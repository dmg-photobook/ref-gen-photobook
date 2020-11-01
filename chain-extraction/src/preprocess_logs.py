import json
import os
import pickle
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

    # Go through all messages in the logs to build vocabulary
    i = 0
    for root, _, files in os.walk(dir_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                i += 1
                if i > 10:
                    break
                # Load game log
                with open(os.path.join(root, file), 'r') as logfile:
                    log = json.load(logfile)

                    # each game log has 5 rounds
                    for round in log['rounds']:

                        # each round has multiple messages
                        for message in round['messages']:

                            # filter out special messages
                            if message['message'].startswith('<selection>'):
                                continue
                            elif message['message'].startswith('<feedback>'):
                                continue
                            elif message['message'].startswith('<next_round>'):
                                continue
                            elif message['message'].startswith('<usr_feedback>'):
                                continue

                            for tok in tokenizer(message['message']):
                                vocab[tok.text] += 1

    return vocab




if __name__ == '__main__':

    f_out = open('../data/logs/all_logs_spellchecked.json', 'w')    #

    logs = {}
    for root, _, files in os.walk('../data/logs'):
        for file in tqdm(files):
            if file.endswith('_spellchecked.json') and not file.startswith('all_logs'):
                with open(os.path.join(root, file), 'r') as logfile:
                    for line in logfile:
                        f_out.write(line)

    f_out.close()

    # tokenizer = spacy.load('en_core_web_sm')
    # lemmatizer = WordNetLemmatizer()
    #
    # en_vocab = set(nltk_en_vocab.words())
    #
    # # vocab = build_vocabulary('../data/logs/')
    # #
    # # with open('../data/logs/vocab.tsv', 'w') as f:
    # #     for w, c in vocab.most_common():
    # #         print('{}\t{}'.format(c, w), file=f)
    #
    # vocab = []
    # counts = []
    # with open('../data/logs/vocab.tsv', 'r') as f:
    #     for line in f:
    #         line = line.rstrip('\n')
    #         c, w = line.split('\t')
    #         vocab.append(w)
    #         counts.append(c)
    #
    # with open('../data/logs/vocab_mispelled.tsv', 'w') as f:
    #     for w, c in zip(vocab, counts):
    #         if w.lower() in en_vocab:
    #             continue
    #         if lemmatizer.lemmatize(w.lower()) in en_vocab:
    #             continue
    #         if w.isnumeric():
    #             continue
    #
    #         print('{}\t{}'.format(c, w), file=f)
    #
    # spell = SpellChecker()
    #
    # with open('../data/logs/vocab_mispelled.tsv', 'r') as f_in:
    #     n_lines = 0
    #     for line in f_in:
    #         n_lines += 1
    #
    # with open('../data/logs/vocab_mispelled.tsv', 'r') as f_in:
    #     with open('../data/logs/vocab_corrected.tsv', 'w') as f_out:
    #
    #         for line in tqdm(f_in, total=n_lines):
    #
    #             line = line.rstrip('\n')
    #             c, w = line.split('\t')
    #
    #             correction = spell.correction(w)
    #             if correction != w:
    #                 print('{}\t{}\t{}'.format(c, w, correction), file=f_out)



# # =================================================================
#     with open('../data/logs/gold_logs.dict', 'rb') as f:
#         gold_logs = pickle.load(f)
#
#     with open('../data/logs/test_logs.dict', 'rb') as f:
#         test_logs = pickle.load(f)
#
#     for game_id in tqdm(test_logs):
#         for round in gold_logs[game_id].rounds:
#             for message in round.messages:
#                 for message in round.messages:
#
#                     # filter out special messages
#                     if message.text.startswith('<selection>'):
#                         continue
#                     elif message.text.startswith('<feedback>'):
#                         continue
#                     elif message.text.startswith('<next_round>'):
#                         continue
#                     elif message.text.startswith('<usr_feedback>'):
#                         continue
#
#                     for tok in tokenizer(message.text):
#                         vocab[tok.text] += 1