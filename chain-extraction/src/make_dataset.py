import os
import json
import sys
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Split chains and compute dataset statistics.')
parser.add_argument('path_all_chains', type=str,
                    help='Path to JSON file containing all extracted chains.')
parser.add_argument('path_gold_chains', type=str,
                    help='Path to JSON file containing gold chains only.')
parser.add_argument('path_output_folder', type=str,
                    help='Path to output folder where the dataset will be stored.')
parser.add_argument('--path_data_splits', type=str, default='data/data_splits.json',
                    help='Path to JSON file containing dataset splits.')
parser.add_argument('--path_domain_ids', type=str, default='data/domain_ids.txt',
                    help='Path to txt file containing domain ids of MSCOCO images.')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Whether to overwrite the content of the output folder.')
args = parser.parse_args()

if os.path.exists(args.path_output_folder):
    if not args.overwrite:
        print('Output folder already exists. If you want to overwrite this folder, use the --overwrite argument.')
        sys.exit()
else:
    os.makedirs(args.path_output_folder)

print('Load chains.')
with open(args.path_gold_chains, 'r') as f:
    gold_chains = json.load(f)

with open(args.path_all_chains, 'r') as f:
    all_chains = json.load(f)

# Correct indexing error (i+1) for round numbers if necessary
need_round_nr_correction = False
for img, img_chains in all_chains.items():
    for game_id, chain in img_chains.items():
        for utt in chain:
            if utt['Round_Nr'] > 5:
                need_round_nr_correction = True
                break
print('Round number correction {}necessary.'.format('' if need_round_nr_correction else 'not '))

if need_round_nr_correction:
    for img, img_chains in all_chains.items():
        for game_id, chain in img_chains.items():
            for utt in chain:
                utt['Round_Nr'] = utt['Round_Nr'] - 1
    # update chain file
    with open(args.path_all_chains, 'w') as f_out:
        json.dump(all_chains, fp=f_out, indent=2, default=str)
    print('Round numbers corrected. Updated {}'.format(args.path_all_chains))
need_round_nr_correction = False


# Obtain splits
with open(args.path_data_splits, 'r') as f:
    splits = json.load(f)

# add dev games to train set
splits = {
    'train': splits['train'] + splits['dev'],
    'test': splits['test'],
    'val': splits['val']
}

# to which split does a game belong?
game2split = {}
for split, games in splits.items():
    for game in games:
        game2split[game] = split

# Split dataset
train_chains = defaultdict(lambda: defaultdict(list))
val_chains = defaultdict(lambda: defaultdict(list))
test_chains = defaultdict(lambda: defaultdict(list))
test_chains_fromgold = defaultdict(lambda: defaultdict(list))
test_chains_gold = defaultdict(lambda: defaultdict(list))
test_chains_nogold = defaultdict(lambda: defaultdict(list))

print('Split dataset.\n')
for img, img_chains in all_chains.items():
    for game_id, chain in img_chains.items():
        try:
            _ = gold_chains[img][game_id]
            test_chains_fromgold[img][int(game_id)] = all_chains[img][game_id]
            test_chains_gold[img][int(game_id)] = gold_chains[img][game_id]
            test_chains[img][int(game_id)] = all_chains[img][game_id]
            continue
        except KeyError:
            pass

        if game2split[int(game_id)] == 'train':
            train_chains[img][int(game_id)] = all_chains[img][game_id]
        elif game2split[int(game_id)] == 'test':
            test_chains[img][int(game_id)] = all_chains[img][game_id]
            test_chains_nogold[img][int(game_id)] = all_chains[img][game_id]
        elif game2split[int(game_id)] == 'val':
            val_chains[img][int(game_id)] = all_chains[img][game_id]


# Count number of images and image domains in each split
print('Number of target images per split:')
print('Train', len(train_chains))
print('Validation', len(val_chains))
print('Test', len(test_chains_nogold))
print('Gold-20', len(test_chains_gold))
print('Extracted-20', len(test_chains_fromgold), '\n')

id2domain = {}
with open(args.path_domain_ids, 'r') as f:
    for line in f.readlines():
        line = line.split(':')
        id_ = int(line[0])
        domain_ = line[1].strip()
        id2domain[id_] = domain_

domains_train = set()
for _, img_chains in train_chains.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_train.add(id2domain[utt['Game_Domain_ID']])

domains_test = set()
for _, img_chains in test_chains.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_test.add(id2domain[utt['Game_Domain_ID']])

domains_val = set()
for _, img_chains in val_chains.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_val.add(id2domain[utt['Game_Domain_ID']])

domains_test_gold = set()
for _, img_chains in test_chains_gold.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_test_gold.add(id2domain[utt['Game_Domain_ID']])

domains_test_fromgold = set()
for _, img_chains in test_chains_fromgold.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_test_fromgold.add(id2domain[utt['Game_Domain_ID']])

domains_test_nogold = set()
for _, img_chains in test_chains_nogold.items():
    for _, chain in img_chains.items():
        for utt in chain:
            domains_test_nogold.add(id2domain[utt['Game_Domain_ID']])

print('Number of image domains per split:')
print('Train', len(domains_train))
print('Validation', len(domains_val))
print('Test', len(domains_test_nogold))
print('Gold-20', len(domains_test_gold))
print('Extracted-20', len(domains_test_fromgold), '\n')


# Count number of chains and number of utterances in each split
n_utt_train, n_chains_train = 0, 0
for _, img_chains in train_chains.items():
    for _, chain in img_chains.items():
        n_utt_train += len(chain)
        n_chains_train += 1

n_utt_test, n_chains_test = 0, 0
for _, img_chains in test_chains.items():
    for _, chain in img_chains.items():
        n_utt_test += len(chain)
        n_chains_test += 1

n_utt_val, n_chains_val = 0, 0
for _, img_chains in val_chains.items():
    for _, chain in img_chains.items():
        n_utt_val += len(chain)
        n_chains_val += 1

n_utt_test_gold, n_chains_test_gold = 0, 0
for _, img_chains in test_chains_gold.items():
    for _, chain in img_chains.items():
        n_utt_test_gold += len(chain)
        n_chains_test_gold += 1

n_utt_test_from_gold, n_chains_test_from_gold = 0, 0
for _, img_chains in test_chains_fromgold.items():
    for _, chain in img_chains.items():
        n_utt_test_from_gold += len(chain)
        n_chains_test_from_gold += 1

n_utt_test_nogold, n_chains_test_nogold = 0, 0
for _, img_chains in test_chains_nogold.items():
    for _, chain in img_chains.items():
        n_utt_test_nogold += len(chain)
        n_chains_test_nogold += 1

Ns_chains = [n_chains_train, n_chains_val, n_chains_test]
Ns = [n_utt_train, n_utt_val, n_utt_test]
tot_utterances = sum(Ns)
tot_chains = sum(Ns_chains)

print('Number of chains per split (and percentage with respect to all chains):')
print('Train', n_chains_train, '({:.2f} %)'.format(100 * n_chains_train / tot_chains))
print('Validation', n_chains_val, '({:.2f} %)'.format(100 * n_chains_val / tot_chains))
print('Test', n_chains_test_nogold, '({:.2f} %)'.format(100 * n_chains_test_nogold / tot_chains))
print('Gold-20', n_chains_test_gold, '({:.2f} %)'.format(100 * n_chains_test_gold / tot_chains))
print('Extracted-20', n_chains_test_from_gold, '({:.2f} %)'.format(100 * n_chains_test_from_gold / tot_chains))
print('Total number of chains: {}\n'.format(tot_chains))

print('Number of utterances per split (and percentage with respect to all utterances):')
print('Train', n_utt_train, '({:.2f} %)'.format(100 * n_utt_train / tot_utterances))
print('Validation', n_utt_val, '({:.2f} %)'.format(100 * n_utt_val / tot_utterances))
print('Test', n_utt_test_nogold, '({:.2f} %)'.format(100 * n_utt_test_nogold / tot_utterances))
print('Gold-20', n_utt_test_gold, '({:.2f} %)'.format(100 * n_utt_test_gold / tot_utterances))
print('Extracted-20', n_utt_test_from_gold, '({:.2f} %)'.format(100 * n_utt_test_from_gold / tot_utterances))
print('Total number of utterances: {}\n'.format(tot_utterances))


# Save splits
if args.path_output_folder[-1] == '/':
    path_output = args.path_output_folder[:-1]

print('Save splits to {}\n'.format(args.path_output_folder))

with open('{}/train.json'.format(args.path_output_folder), 'w') as f_out:
    json.dump(train_chains, fp=f_out, indent=2, default=str)

with open('{}/val.json'.format(args.path_output_folder), 'w') as f_out:
    json.dump(val_chains, fp=f_out, indent=2, default=str)

with open('{}/test.json'.format(args.path_output_folder), 'w') as f_out:
    json.dump(test_chains_nogold, fp=f_out, indent=2, default=str)

with open('{}/gold-extracted.json'.format(args.path_output_folder), 'w') as f_out:
    json.dump(test_chains_fromgold, fp=f_out, indent=2, default=str)

with open('{}/gold.json'.format(args.path_output_folder), 'w') as f_out:
    json.dump(test_chains_gold, fp=f_out, indent=2, default=str)


# Compute chain length and utterance length statistics
print('Chain length and utterance length statistics.')
# all chains
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in all_chains:
    for game in all_chains[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(all_chains[img][game]))
        for utterance in all_chains[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('ALL CHAINS. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('ALL CHAINS. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('ALL CHAINS. Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))

# training chains
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in train_chains:
    for game in train_chains[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(train_chains[img][game]))
        for utterance in train_chains[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('TRAIN. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('TRAIN. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('TRAIN. Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))

# validation chains
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in val_chains:
    for game in val_chains[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(val_chains[img][game]))
        for utterance in val_chains[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('VALIDATION. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('VALIDATION. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('VALIDATION: Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))

# test chains
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in test_chains_nogold:
    for game in test_chains_nogold[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(test_chains_nogold[img][game]))
        for utterance in test_chains_nogold[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('TEST. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('TEST. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('TEST: Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))

# gold-20
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in test_chains_gold:
    for game in test_chains_gold[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(test_chains_gold[img][game]))
        for utterance in test_chains_gold[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('GOLD-20. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('GOLD-20. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('GOLD-20: Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))

# extracted-20
chain_lengths = []
utterance_lengths = []
unique_utterances = defaultdict(int)
for img in test_chains_fromgold:
    for game in test_chains_fromgold[img]:
        # the number of utterances in a chain
        chain_lengths.append(len(test_chains_fromgold[img][game]))
        for utterance in test_chains_fromgold[img][game]:
            # the number of tokens in an utterance
            utterance_lengths.append(len(utterance['Message_Text'].split(' ')))
            unique_utterances[utterance['Message_Text']] += 1

print('EXTRACTED-20. Chain length:')
print('min: {:.2f}'.format(np.min(chain_lengths)))
print('max: {:.2f}'.format(np.max(chain_lengths)))
print('median: {:.2f}'.format(np.median(chain_lengths)))
print('mean: {:.2f}'.format(np.mean(chain_lengths)))
print('std: {:.2f}'.format(np.std(chain_lengths)))
print('% 1 utt: {:.2f}'.format(100 * (len([l for l in chain_lengths if l == 1]) / len(chain_lengths))))
print('EXTRACTED-20. Utterance length:')
print('min: {:.2f}'.format(np.min(utterance_lengths)))
print('max: {:.2f}'.format(np.max(utterance_lengths)))
print('median: {:.2f}'.format(np.median(utterance_lengths)))
print('mean: {:.2f}'.format(np.mean(utterance_lengths)))
print('std: {:.2f}'.format(np.std(utterance_lengths)))
print('EXTRACTED-20: Unique utterances: {} out of {}.\n'.format(
    len(unique_utterances), sum(list(unique_utterances.values()))))


# Compute first vs. later utterance statistics
print('First vs. Later utterances.')

# all chains
first_len, later_len = [], []
for img in all_chains:
    for game in all_chains[img]:
        for i, utterance in enumerate(all_chains[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('ALL CHAINS. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('ALL CHAINS. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})\n'.format(np.mean(later_len), np.std(later_len)))


# train
first_len, later_len = [], []
for img in train_chains:
    for game in train_chains[img]:
        for i, utterance in enumerate(train_chains[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('TRAIN. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('TRAIN. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})\n'.format(np.mean(later_len), np.std(later_len)))

# validation
first_len, later_len = [], []
for img in val_chains:
    for game in val_chains[img]:
        for i, utterance in enumerate(val_chains[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('VALIDATION. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('VALIDATION. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})\n'.format(np.mean(later_len), np.std(later_len)))

# test
first_len, later_len = [], []
for img in test_chains_nogold:
    for game in test_chains_nogold[img]:
        for i, utterance in enumerate(test_chains_nogold[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('TEST. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('TEST. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})\n'.format(np.mean(later_len), np.std(later_len)))

# gold-20
first_len, later_len = [], []
for img in test_chains_gold:
    for game in test_chains_gold[img]:
        for i, utterance in enumerate(test_chains_gold[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('GOLD-20. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('GOLD-20. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})\n'.format(np.mean(later_len), np.std(later_len)))

# extracted-20
first_len, later_len = [], []
for img in test_chains_fromgold:
    for game in test_chains_fromgold[img]:
        for i, utterance in enumerate(test_chains_fromgold[img][game]):
            if i == 0:
                first_len.append(len(utterance['Message_Text'].split(' ')))
            else:
                later_len.append(len(utterance['Message_Text'].split(' ')))
print('EXTRACTED-20. Number of utterances:')
print('First: {}. Later: {}.'.format(len(first_len), len(later_len)))
print('EXTRACTED-20. Average utterance length and standard deviation:')
print('First: {:.2f} ({:.2f})'.format(np.mean(first_len), np.std(first_len)))
print('Later: {:.2f} ({:.2f})'.format(np.mean(later_len), np.std(later_len)))
