import argparse
import json
import pickle
from copy import deepcopy
from collections import defaultdict
from utils import load_logs, group_by_game


def extract(logs, from_first_common=True, first_ref_only=False):
    chains = defaultdict(list)

    for game_id, log in logs.items():

        tracked_in_game = set()

        for game_round in log.rounds:

            buffer = []

            for message in game_round.messages:

                if message.type == 'selection':
                    # parse selection: <com>/<dif> + img_id_str
                    _, selection, img_path = message.text.split(' ')

                    if selection == '<com>' and img_path in game_round.common:
                        tracked_in_game.add(img_path)

                    elif selection == '<dif>' and not from_first_common:
                        tracked_in_game.add(img_path)

                if message.type == 'text':
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
                                 'Round Duration': game_round.duration,
                                 'In_Segment': True,
                                 'Reason': '<gold>',
                                 'Meteor_Score': 10,
                                 'Precision_Score': 10,
                                 'Recall_Score': 10,
                                 'F1_Score': 10,
                                 'score': 10,
                                 'Discriminative_Features': {},
                                 'All_Features': {},
                                 'Message_Referent': message.referent
                                 }
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
                for _idx, utterance in enumerate(buffer):
                    tmp_utterance = deepcopy(utterance)
                    speaker_img_set = tmp_utterance['Round_Images_{}'.format(tmp_utterance['Message_Speaker'])]

                    if tmp_utterance['Message_Referent'] == img and img in speaker_img_set:
                        tmp_buffer.append(tmp_utterance)
                        if first_ref_only:
                            break

                if tmp_buffer:
                    chains[img] += tmp_buffer

    return chains


def main(logs_path,
         output_path,
         from_first_common,
         first_reference_only):

    if not (output_path.endswith('.dict') or output_path.endswith('.json')):
        raise ValueError('Invalid output path:', output_path)

    # Load game logs
    if logs_path.endswith('.pickle') or logs_path.endswith('.dict'):
        with open(logs_path, 'rb') as f:
            gold_logs = pickle.load(f)  # e.g. gold_logs.dict
    else:
        gold_logs = load_logs(logs_path)

    print('>> Extract segments from gold logs')
    gold_segments = extract(gold_logs, from_first_common=from_first_common, first_ref_only=first_reference_only)
    chains = group_by_game(gold_segments)

    # Store segments as chains: irrelevant utterances are excluded by default
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f_out:
            json.dump(chains, fp=f_out, indent=2, default=str)
    else:
        with open(output_path, 'wb') as f_out:
            pickle.dump(chains, file=f_out)

    print('>> Gold chains saved to: {} \n'.format(output_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build reference chains from annotated PhotoBook game logs.')
    parser.add_argument('path_output', type=str,
                        help='Path to output JSON file which will contain the gold reference chains.')
    parser.add_argument('--path_game_logs', type=str, default='data/logs/gold_logs.dict',
                        help='Path to the annotated game logs.')
    parser.add_argument('--from_first_common', action='store_true',
                        help='Whether to start collecting referring utterances only after the target image has been '
                             'seen by both participants.')
    parser.add_argument('--first_reference_only', action='store_true',
                        help='Whether to only collect the first referring expression in the round for a given target'
                             'image. Alternatively, all referring expressions in the round are collected')
    args = parser.parse_args()

    FIELDS = ['Agent_1',
              'Agent_2',
              'Feedback_A',
              'Feedback_B',
              'Game, Duration',
              'Game_Domain_1',
              'Game_Domain_2',
              'Game_Domain_ID',
              'Game_ID',
              # 'Game_Scores',
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
              # 'Round_Scores',
              # 'Total_Game_Score',
              # 'Total_Round_Score'
              ]

    main(logs_path=args.path_game_logs, output_path=args.path_output,
         from_first_common=args.from_first_common, first_reference_only=args.first_reference_only)
