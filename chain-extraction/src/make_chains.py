import argparse
import json
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
from utils import group_by_game


def filter(segments_path, score='PrecisionScore', nbest=4):
    with open(segments_path, 'rb') as f:
        segments = pickle.load(file=f)

    chains = {img: [] for img in segments}

    print('>> Filter out irrelevant utterances using {}'.format(score))

    selected_utterances = defaultdict(list)  # (game, round) -> [(utt, img), ...]

    for img, img_segments in tqdm(segments.items()):

        nbest_utterances = [{'Message_Text': '', score: 0, 'Message_Referent': None} for _ in range(nbest)]
        current_round = (-1, -1)

        for utterance in img_segments:

            if current_round != (int(utterance['Game_ID']), utterance['Round_Nr']):

                candidates = []
                for candidate in nbest_utterances:
                    if candidate[score] > 0:
                        candidates.append(candidate)

                        key = (candidate['Game_ID'], candidate['Round_Nr'])
                        selected_utterances[key].append((candidate, img))

                if candidates:
                    key0 = (candidates[0]['Game_ID'], candidates[0]['Round_Nr'])
                    for c in candidates:
                        assert (c['Game_ID'], c['Round_Nr']) == key0

                    chains[img].append((key0, candidates))

                # reset
                current_round = (utterance['Game_ID'], utterance['Round_Nr'])
                nbest_utterances = [{'Message_Text': '', score: 0, 'Message_Referent': None} for _ in range(nbest)]

            # Only consider utterances by the participant who sees the target image
            speaker_img_set = utterance['Round_Images_{}'.format(utterance['Message_Speaker'])]

            if utterance['In_Segment'] and img in speaker_img_set:

                for j in range(nbest):
                    if utterance[score] >= nbest_utterances[j][score]:
                        nbest_utterances.insert(j, utterance)
                        nbest_utterances.pop()
                        break

    # cnt = Counter()
    # for img in chains:
    #     for key, ulist in chains[img]:
    #         cnt[len(ulist)] += 1
    # print(cnt)
    # print(sum(list(cnt.values())))

    chains_tmp = {img: [] for img in chains}

    for img in chains:
        for key, candidates in chains[img]:
            candidates_updated = []

            for candidate in candidates:
                keep_candidate = True

                for (u, i) in selected_utterances[key]:
                    # if candidate has already been selected in this game and round for another image
                    if u['Message_Text'] == candidate['Message_Text'] and i != img:

                        if candidate[score] <= u[score]:
                            keep_candidate = False
                            break

                if keep_candidate:
                    candidates_updated.append(candidate)

            chains_tmp[img].append(candidates_updated)

    # cnt = Counter()
    # for img in chains_tmp:
    #     for ulist in chains_tmp[img]:
    #         cnt[len(ulist)] += 1
    # print(cnt)
    # print(sum(list(cnt.values())))

    chains_final = {img: [] for img in chains}
    for img in chains_tmp:
        for ulist in chains_tmp[img]:
            if ulist:
                chains_final[img].append(ulist[0])

    return chains_final


def main(segments_path, output_path, score='Precision_Score'):
    if not (output_path.endswith('.dict') or output_path.endswith('.json') or output_path.endswith('.tsv')):
        raise ValueError('Invalid output path:', output_path)

    # Obtain chains by filtering out utterances that are unrelated to the target image
    chains = filter(segments_path, score=score, nbest=4)
    chains = group_by_game(chains)

    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(chains, fp=f, indent=2, default=str)
    elif output_path.endswith('.tsv'):
        with open(output_path, 'w') as f:
            k0 = list(chains.keys())[0]
            utt0 = chains[k0][0]
            print('\t'.join(['Target'] + [k for k in utt0.keys()]), file=f)
            for img, img_chains in chains.items():
                for utterance in img_chains:
                    print('\t'.join([img] + [str(v) for v in utterance.values()]), file=f)
    else:
        with open(output_path, 'wb') as f:
            pickle.dump(chains, file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build reference chains from extracted segments.')
    parser.add_argument('path_segments', type=str, help='Path to the .dict file containing the extracted segments.')
    parser.add_argument('path_output', type=str, help='Path to output JSON file which will contain the extracted '
                                                      'reference chains.')
    parser.add_argument('--score', type=str, default='f1', help='Scoring function: f1, recall, precision')
    args = parser.parse_args()

    score_fn = args.score.lower()
    if score_fn == 'f1':
        score_fn = 'F1_Score'
    elif score_fn == 'recall':
        score_fn = 'Recall_Score'
    elif score_fn == 'precision':
        score_fn = 'Precision_Score'
    else:
        raise ValueError('Invalid --score argument.')

    main(args.path_segments, args.path_output, score=score_fn)
