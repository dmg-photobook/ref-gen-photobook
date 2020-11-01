import json
import os
import pickle
from json import JSONDecodeError

from nltk.corpus import words as nltk_en_vocab
import spacy
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tqdm import tqdm
from spellchecker import SpellChecker

def build_vocabulary(dir_path):

    # Safefy check: count logs
    file_count = 0
    for _, _, files in os.walk(dir_path):
        for file in files:
            file_count += int(file.endswith('.json'))
    print('{} logs found.'.format(file_count))

    vocab = Counter()




    # # Go through all messages in the logs to build vocabulary
    # i = 0
    # for root, _, files in os.walk(dir_path):
    #     for file in tqdm(files):
    #         if file.endswith('.json'):
    #             i += 1
    #             if i > 10:
    #                 break
    #             # Load game log
    #             with open(os.path.join(root, file), 'r') as logfile:
    #                 log = json.load(logfile)
    #
    #                 # each game log has 5 rounds
    #                 for round in log['rounds']:
    #
    #                     # each round has multiple messages
    #                     for message in round['messages']:
    #
    #                         # filter out special messages
    #                         if message['message'].startswith('<selection>'):
    #                             continue
    #                         elif message['message'].startswith('<feedback>'):
    #                             continue
    #                         elif message['message'].startswith('<next_round>'):
    #                             continue
    #                         elif message['message'].startswith('<usr_feedback>'):
    #                             continue
    #
    #                         for tok in tokenizer(message['message']):
    #                             vocab[tok.text] += 1
    #
    # return vocab




if __name__ == '__main__':

    typo_corrections = {}
    with open('../data/logs/typos.tsv', 'r') as f_in:
        for line in f_in:
            line = line.rstrip('\n')
            try:
                freq, typo, correction, correct = line.split('\t')
                freq = int(freq)
                correct = bool(int(correct))
            except ValueError:
                _, _, _ = line.split('\t')
                continue

            if correct:
                typo_corrections[typo] = (correction, freq)

    print(typo_corrections)
    tokenizer = spacy.load('en_core_web_sm')

    i = 0
    for root, _, files in os.walk('../data/logs'):
        for file in tqdm(files):
            if file.endswith('.json') and not file.endswith('_spellchecked.json') and not file.startswith('all_logs'):
                # i += 1
                # if i > 10:
                #     break

                # Load game log
                with open(os.path.join(root, file), 'r') as logfile_in:

                    try:
                        log = json.load(logfile_in)
                    except JSONDecodeError:
                        print(file)

                    # each game log has 5 rounds
                    for rnr, round in enumerate(log['rounds']):

                        # each round has multiple messages
                        for mnr, message in enumerate(round['messages']):

                            # filter out special messages
                            try:
                                if message['message'].startswith('<selection>'):
                                    continue
                                elif message['message'].startswith('<feedback>'):
                                    continue
                                elif message['message'].startswith('<next_round>'):
                                    continue
                                elif message['message'].startswith('<usr_feedback>'):
                                    continue
                            except AttributeError:
                                print('ERROR:::', message)
                                i += 1

                            # new_msg = []
                            typos = []
                            # print(message['message'])

                            # if isinstance(message['message'], str):
                            #     tokens = tokenizer(message['message'])
                            # elif isinstance(message['message'], list):
                            #     tokens = message['message']
                            # else:
                            #     raise ValueError('Message:', message)

                            for tok in tokenizer(message['message']):
                                tok = tok.text
                                if tok in typo_corrections:
                                    # print('typo:', tok)
                                    # new_msg.append(typo_corrections[tok][0])
                                    typos.append(tok)
                                    # try:
                                    typo_corrections[tok] = (typo_corrections[tok][0], typo_corrections[tok][1] - 1)
                                    # except TypeError:
                                    #     print(typo_corrections[tok])
                                    assert typo_corrections[tok][1] >= 0
                                # else:
                                #     new_msg.append(tok)
                            # print(typos)
                            # if typos:
                            #     print('1', message['message'])

                            for typo in typos:
                                # print('typo', typo, typo_corrections[typo][0])
                                log['rounds'][rnr]['messages'][mnr]['message'] = message['message'].replace(typo, typo_corrections[typo][0])
                                # print(message['message'])
                            # if typos:
                            #     print('2', message['message'])

                    with open(os.path.join(root, file[:-len('.json')] + '_spellchecked.json'), 'w') as logfile_out:
                        json.dump(log, fp=logfile_out, indent=2, default=str)

    print(i)