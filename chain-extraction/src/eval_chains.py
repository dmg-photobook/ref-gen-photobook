import argparse
import os
import json
import pickle
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, Counter


def eval(_chains, score='F1_Score', strict_recall=False):
    extracted_chains = defaultdict(list)

    true_pos, false_pos, false_neg = 0, 0, 0
    true_pos_samples, false_pos_samples, false_neg_samples = defaultdict(list), defaultdict(list), defaultdict(list)

    chains = {img: [] for img in _chains}
    selected_utterances = defaultdict(list)  # (game, round) -> [(utt, img), ...]
    nbest = 4

    gold_positive = defaultdict(list)  #  (game, round, img) -> [utt1, ...]

    for img, img_chains in _chains.items():

        nbest_utterances = [{'Message_Text': '', score: 0, 'Message_Referent': None} for _ in range(nbest)]
        current_round = (-1, -1)

        for utterance in img_chains:
            if utterance['Message_Referent'] == img:
                gold_positive[(utterance['Game_ID'], utterance['Round_Nr'], img)].append(utterance)

            if current_round != (utterance['Game_ID'], utterance['Round_Nr']):

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

            speaker_img_set = utterance['Round_Images_{}'.format(utterance['Message_Speaker'])]

            if utterance['In_Segment'] and img in speaker_img_set:

                for j in range(nbest):
                    if utterance[score] >= nbest_utterances[j][score]:
                        nbest_utterances.insert(j, utterance)
                        nbest_utterances.pop()
                        break

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

            chains_tmp[img].append((key, candidates_updated))

    for img in chains_tmp:
        for key, ulist in chains_tmp[img]:
            best_utt = None
            if ulist:
                best_utt = ulist[0]

                if best_utt['Message_Referent'] == img:
                    true_pos += 1
                    true_pos_samples[img].append(best_utt)
                else:
                    false_pos += 1
                    false_pos_samples[img].append(best_utt)

            for gold_pos_utt in gold_positive[(key[0], key[1], img)]:
                if best_utt is None or best_utt['Message_Text'] != gold_pos_utt['Message_Text']:
                    false_neg_samples[img].append(gold_pos_utt)
                    false_neg += 1
                    if not strict_recall:
                        break

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    return precision, recall, true_pos_samples, false_pos_samples, false_neg_samples, extracted_chains


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate reference chains.')
    parser.add_argument('path_segments', type=str, help='Path to the segments extracted from annotated game logs.')
    args = parser.parse_args()

    with open(args.path_segments, 'rb') as f:
        chains_ = pickle.load(file=f)

        for score in ['Precision_Score', 'Recall_Score', 'F1_Score']:
            print('Heuristic:', score)
            P, R, _, _, _, _ = eval(chains_, score)
            print('Precision: {:.2f}'.format(P))
            print('Recall: {:.2f}'.format(R))
