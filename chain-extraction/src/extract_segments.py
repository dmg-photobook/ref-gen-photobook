import argparse
import json
import pickle
import spacy
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from nltk.translate.meteor_score import single_meteor_score as meteor
from bertscore import mean_bert_precision, mean_bert_f1, mean_bert_recall
from utils import text_to_bow, stopwords_filter, load_logs, preprocess_captions, get_captions


def get_best_description(utterances):
    best_description = ('', [], None, 0)
    for utterance in utterances:
        if utterance[3] >= best_description[3]:
            best_description = utterance

    description = best_description[0]
    description_as_bow = best_description[2]

    if best_description[3] <= 0:
        return None
    else:
        return tuple(zip(description, description_as_bow))


def vg_score(text, img_path, vg_attributes, vg_relations, visual_context):
    """
    METEOR score based on Visual genome entity and relationship tokens.
    Lavie and Agarwal, 2007 (http://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf)
    """
    target_attributes = vg_attributes[img_path]
    target_relations = vg_relations[img_path]

    all_features = target_attributes | target_relations

    visual_context -= {img_path}

    confounding_attributes = set()
    confounding_relations = set()
    for path in visual_context:
        confounding_attributes |= vg_attributes[path]
        confounding_relations |= vg_relations[path]

    discriminative_attributes = target_attributes - confounding_attributes
    discriminative_relations = target_relations - confounding_relations

    discriminative_features = discriminative_attributes | discriminative_relations
    meteor_score = meteor(' '.join(discriminative_features), text, gamma=0)

    return meteor_score, discriminative_features, all_features


def extract(logs, from_first_common=True):
    segments = defaultdict(list)

    for game_id, log in tqdm(logs.items()):

        tracked_in_game = set()

        for game_round in log.rounds:

            buffer = []
            metadata = defaultdict(lambda: {'last_index': 0, 'reason': ''})

            for message in game_round.messages:

                if message.type == 'selection':
                    # parse selection: <com>/<dif> + img_id_str
                    _, selection, img_path = message.text.split(' ')

                    if img_path in tracked_in_game:
                        metadata[img_path]['last_index'] = len(buffer)
                        metadata[img_path]['reason'] = selection

                    elif selection == '<com>' and img_path in game_round.common:
                        tracked_in_game.add(img_path)
                        metadata[img_path]['last_index'] = len(buffer)
                        metadata[img_path]['reason'] = selection

                    elif selection == '<dif>':
                        tracked_in_game.add(img_path)
                        metadata[img_path]['reason'] = selection
                        if from_first_common:
                            metadata[img_path]['last_index'] = 0
                        else:
                            metadata[img_path]['last_index'] = len(buffer)

                elif message.type == 'text':
                    utterance = {'Game_ID': game_id, 'Round_Nr': game_round.round_nr, 'Message_Nr': message.message_id,
                                 'Message_Speaker': message.speaker, 'Message_Type': message.type,
                                 'Message_Text': message.text, 'Round_Common': game_round.common,
                                 'Round_Images_A': game_round.images['A'], 'Round_Images_B': game_round.images['B'],
                                 'Game_Domain_ID': log.domain_id,
                                 'Game_Domain_1': log.domains[0],
                                 'Game_Domain_2': log.domains[1], 'Feedback_A': log.feedback['A'],
                                 'Feedback_B': log.feedback['B'], 'Agent_1': log.agent_ids[0],
                                 'Agent_2': log.agent_ids[1], 'Round_Highlighted_A': game_round.highlighted['A'],
                                 'Round_Highlighted_B': game_round.highlighted['B'],
                                 'Message_Timestamp': message.timestamp, 'Message_Turn': message.turn,
                                 'Message_Agent_ID': message.agent_id,
                                 'Game Duration': log.duration,
                                 'N_Messages_In_Round': game_round.num_messages,
                                 'Round Duration': game_round.duration
                                 }
                    try:
                        utterance['Message_Referent'] = message.referent
                    except AttributeError:
                        pass

                    try:
                        utterance['Total_Game_Score'] = log.total_score
                        utterance['Game_Scores'] = {k: v for k, v in log.scores.items()}
                        utterance['Round_Scores'] = game_round.scores
                        utterance['Total_Round_Score'] = game_round.total_score
                    except AttributeError:
                        pass
                    buffer.append(utterance)

            # 1. store utterances from previous round
            for img in tracked_in_game:
                tmp_buffer = []
                any_utterance_in_segment = False
                for _idx, utterance in enumerate(buffer):
                    tmp_utterance = deepcopy(utterance)
                    tmp_utterance['In_Segment'] = _idx < metadata[img]['last_index']
                    tmp_utterance['Reason'] = metadata[img]['reason']
                    tmp_buffer.append(tmp_utterance)
                    any_utterance_in_segment = any_utterance_in_segment or tmp_utterance['In_Segment']

                if any_utterance_in_segment:
                    segments[img] += tmp_buffer

    return segments


def score(image_paths,
          chains,
          caption_representations,
          remove_stopwords,
          remove_nondiscriminative_caption_words,
          descriptions_as_captions,
          vg_attributes=None,
          vg_relations=None):
    # Whether to use Visual Genome scene graphs
    use_vg = (vg_attributes is not None) and (vg_relations is not None)

    _caption_reprs = deepcopy(caption_representations)
    segments_with_scores = defaultdict(list)

    for img_path in tqdm(image_paths):

        # get image id as a string for compatibility
        img_id_str = img_path.split('/')[-1]

        if descriptions_as_captions:
            current_round = -1
            current_game = -1
            utterances_in_current_round = []

        new_c = []
        for fields in chains[img_path]:
            if descriptions_as_captions and fields['Game_ID'] != current_game:
                _caption_reprs = deepcopy(caption_representations)
                current_game = fields['Game_ID']
                current_round = fields['Round_Nr']

            stopwords = set()
            if remove_stopwords:
                stopwords = stopwords_en

            caption_stopwords = set()
            if remove_nondiscriminative_caption_words:
                for full_img_path in set(fields['Round_Images_A']) | set(fields['Round_Images_B']) - {img_path}:
                    for caption_tuple in _caption_reprs[full_img_path.split('/')[-1]]:
                        caption_tokens = [c_tok for c_tok, _, in caption_tuple]
                        caption_stopwords |= set(caption_tokens)

            try:
                # convert text to a bag-of-contextualised-words representations
                tokens, input_ids, utt_bow = text_to_bow(fields['Message_Text'], model, tokenizer, stopwords)
                fields['Tokens'] = tokens
            except TypeError:
                # as a result of stopwords filtering, the text may be empty
                continue

            fields['score'] = -1
            fields['Discriminative_Features'], fields['All_features'] = {}, {}

            if fields['In_Segment']:
                # compute recall using BERTScore (Zhang et al. 2019)
                for score_str, bert_score_fn in zip(['Precision_Score', 'Recall_Score', 'F1_Score'],
                                                    [mean_bert_precision, mean_bert_recall, mean_bert_f1]):
                    fields[score_str] = bert_score_fn(_caption_reprs[img_id_str], utt_bow,
                                                      stopwords=(stopwords | caption_stopwords))

                if use_vg:
                    visual_context = set(fields['Round_Images_A']) | set(fields['Round_Images_B'])

                    # preprocess text
                    text_for_vg = ' '.join([tok.text for tok in spacy_tokenizer(fields['Message_Text'])])
                    if remove_stopwords:
                        text_for_vg = stopwords_filter(text_for_vg, stopwords_en)

                    # compute METEOR using Visual Genome annotations
                    meteor_score, discriminative_features, all_features = vg_score(text_for_vg, img_path,
                                                                                   vg_attributes,
                                                                                   vg_relations,
                                                                                   visual_context)
                    fields['Meteor_Score'] = meteor_score
                    fields['Precision_Score'] += meteor_score
                    fields['Recall_Score'] += meteor_score
                    fields['F1_Score'] += meteor_score

                    fields['Discriminative_Features'] = discriminative_features
                    fields['All_features'] = all_features

                fields['score'] = fields['Precision_Score']

                if descriptions_as_captions and fields['score'] > 0:
                    utterances_in_current_round.append((tokens, input_ids[0][1:-1].numpy(), utt_bow, fields['score']))

            # base case - first iteration
            if descriptions_as_captions and current_round == -1:
                current_round = fields['Round_Nr']

            # new round
            if descriptions_as_captions and current_round != fields['Round_Nr']:
                best_description_in_round = get_best_description(utterances_in_current_round)
                if best_description_in_round:
                    _caption_reprs[img_id_str].append(best_description_in_round)

                current_round = fields['Round_Nr']
                utterances_in_current_round = []

            # store current utterance with score
            new_c.append(fields)

        # store current image's chains_3feb (with scores)
        segments_with_scores[img_path] += new_c

    return segments_with_scores


def main(logs_path,
         output_path,
         remove_stopwords,
         remove_caption_words,
         from_first_common,
         descriptions_as_captions,
         use_vg):

    if not (output_path.endswith('.dict') or output_path.endswith('.json')):
        raise ValueError('Invalid output path:', output_path)

    # Load game logs
    if logs_path.endswith('.pickle') or logs_path.endswith('.dict'):
        with open(logs_path, 'rb') as f:
            all_logs = pickle.load(f)  # e.g. gold_logs.dict
    else:
        all_logs = load_logs(logs_path)

    print('>> Extract segments from logs')
    all_segments = extract(all_logs, from_first_common=from_first_common)

    print('>> Load and encode relevant captions')
    all_captions = get_captions('{}/mscoco/captions_train2014.json'.format(args.path_data), all_segments)

    print('>> Load Visual Genome data')
    if use_vg:
        with open('{}/visual_genome/attributes.dict'.format(args.path_data), 'rb') as f:
            vg_attributes = pickle.load(f)
        with open('{}/visual_genome/relationships.dict'.format(args.path_data), 'rb') as f:
            vg_relations = pickle.load(f)
    else:
        vg_attributes, vg_relations = None, None

    image_paths = list(all_segments.keys())

    print('>> Obtain bag-of-contextualised-words representations for all relevant captions')
    caption_representations = preprocess_captions(image_paths, all_captions, model, tokenizer)

    print('>> Score utterances')
    segments_with_scores = score(
        image_paths, all_segments, caption_representations,
        remove_stopwords=remove_stopwords,
        remove_nondiscriminative_caption_words=remove_caption_words,
        descriptions_as_captions=descriptions_as_captions,
        vg_attributes=vg_attributes,
        vg_relations=vg_relations
    )

    # Store chains with scores
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f_out:
            json.dump(segments_with_scores, fp=f_out, indent=2, default=str)
    else:
        with open(output_path, 'wb') as f_out:
            pickle.dump(segments_with_scores, file=f_out)

    print('>> Chains saved to: {} \n'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract segments of relevant utterances from PhotoBook game logs.')
    parser.add_argument('path_output', type=str,
                        help='Path to output .dict file which will contain the extracted segments')
    parser.add_argument('--path_data', type=str, default='data',
                        help='Path to data folder.')
    parser.add_argument('--path_game_logs', type=str, default='data/logs',
                        help='Path to game logs folder or to the annotated game logs file.')
    parser.add_argument('--stopwords', action='store_true',
                        help='Whether to remove stopwords for the computation of utterances scores.')
    parser.add_argument('--meteor', action='store_true',
                        help='Whether to compute METEOR score using Virtual Genome scene graphs.')
    parser.add_argument('--from_first_common', action='store_true',
                        help='Whether to start collecting referring utterances only after the target image has been '
                             'seen by both participants.')
    parser.add_argument('--utterances_as_captions', action='store_true',
                        help='Whether to use extracted referring utterances as additional reference captions.')
    parser.add_argument('--discriminative_captions', action='store_true',
                        help='Whether to remove non-discriminative words from the MSCOCO captions.')

    args = parser.parse_args()

    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    spacy_tokenizer = spacy.load('en_core_web_sm')

    stopwords_en = spacy.lang.en.stop_words.STOP_WORDS

    stopwords_en |= {'sorry', 'no', 'noo', 'nope', 'oh', 'got', 'ha', '.', '!', '?', ','}
    #stopwords_en |= {'sorry', 'no', 'noo', 'nope', 'yes', 'yeah', 'ok', 'oh', 'got', 'ha', '.', '!', '?', ',', '(', ')'}

    stopwords_en -= {'above', 'across', 'against', 'all', 'almost', 'alone', 'along', 'among', 'amongst', 'at', 'back',
                     'behind', 'below', 'beside', 'between', 'beyond', 'bottom', 'down', 'eight', 'eleven', 'empty',
                     'few', 'fifteen', 'fifty', 'five', 'forty', 'four', 'front', 'full', 'hundred', 'he', 'him', 'his',
                     'himself', 'in', 'into', 'many', 'next', 'nine', 'nobody', 'none', 'noone', 'not', 'off', 'on',
                     'one', 'only', 'onto', 'out', 'over', 'part', 'several', 'side', 'she', 'her', 'herself', 'six',
                     'sixty', 'some', 'someone', 'something', 'somewhere', 'ten', 'their', 'them', 'themselves', 'they',
                     'three', 'through', 'thru', 'together', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
                     'under', 'up', 'used', 'using', 'various', 'very', 'with', 'within', 'without'}

    FIELDS = ['Agent_1',
              'Agent_2',
              'Feedback_A',
              'Feedback_B',
              'Game, Duration',
              'Game_Domain_1',
              'Game_Domain_2',
              'Game_Domain_ID',
              'Game_ID',
              'Game_Scores',
              'Message_Agent_ID',
              'Message_Nr',
              'Message_Speaker',
              'Message_Text',
              'Message_Timestamp',
              'Message_Turn',
              'Message_Type', 'N_Messages_In_Round',
              'Round, Duration',
              'Round_Common',
              'Round_Highlighted_A',
              'Round_Highlighted_B',
              'Round_Images_A',
              'Round_Images_B',
              'Round_Nr',
              'Round_Scores',
              'Total_Game_Score',
              'Total_Round_Score']

    main(logs_path=args.path_game_logs, output_path=args.path_output, remove_stopwords=args.stopwords,
         use_vg=args.meteor, remove_caption_words=args.discriminative_captions,
         from_first_common=args.from_first_common, descriptions_as_captions=args.utterances_as_captions)
