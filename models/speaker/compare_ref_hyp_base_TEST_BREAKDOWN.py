import json
import torch
import argparse
import os

from utils.SpeakerDatasetCopy import SpeakerDataset
from utils.Vocab import Vocab

import numpy as np

from bert_score import score

from nlgeval import NLGEval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="./data")
    parser.add_argument("-utterances_file", type=str, default="ids_utterances.pickle")
    parser.add_argument("-chains_file", type=str, default="text_chains.json")
    parser.add_argument('-orig_ref_file', type=str, default='text_utterances.pickle')
    parser.add_argument("-vocab_file", type=str, default="vocab.csv")
    parser.add_argument("-vectors_file", type=str, default="vectors.json")
    parser.add_argument("-model_type", type=str, default='copy')
    parser.add_argument("-subset_size", type=int, default=-1)  # -1 is the full dataset, if you put 10, it will only use 10 chains
    parser.add_argument("-shuffle", action='store_true')
    parser.add_argument("-normalize", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-embedding_dim", type=int, default=512)
    parser.add_argument("-hidden_dim", type=int, default=512)
    parser.add_argument("-attention_dim", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-print", action='store_true')
    parser.add_argument("-metric", type=str, default='cider')  # some metric or loss
    parser.add_argument("-dropout_prob", type=float, default=0.0)
    parser.add_argument("-reduction", type=str, default='sum')  # reduction for NLL loss
    parser.add_argument("-beam_size", type=int, default=5)
    parser.add_argument("-coverage_weight", type=float, default=1)  # coverage loss weight

    args = parser.parse_args()

    seed = 28
    print('seed', seed)

    # for reproducibilty
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print(args)

    print("Loading the vocab...")
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file))
    vocab.index2word[len(vocab)] = '<nohs>'  # special token placeholder for no prev utt
    vocab.word2index['<nohs>'] = len(vocab)  # len(vocab) updated (depends on w2i)

    print("Loading the full vocab")
    vocab_full = Vocab(os.path.join(args.data_path, 'vocab_COPY.csv'))
    vocab_full.index2word[len(vocab_full)] = '<nohs>'  # special token placeholder for no prev utt
    vocab_full.word2index['<nohs>'] = len(vocab_full)  # len(vocab) updated (depends on w2i)

    testset = SpeakerDataset(
        data_dir=args.data_path,
        utterances_file='test_' + args.utterances_file,
        vectors_file=args.vectors_file,
        chain_file='test_' + args.chains_file,
        orig_ref_file='test_' + args.orig_ref_file,
        split='test',
        subset_size=args.subset_size,
        vocab_obj=vocab,
        actual_vocab_obj=vocab_full
    )

    print('vocab len', len(vocab))
    print('test len', len(testset), 'longest sentence', testset.max_len)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    metric = args.metric
    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    shuffle = args.shuffle

    batch_size = 1

    load_params_test = {'batch_size': 1, 'shuffle': False,
                        'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<sos>'], vocab['<eos>'],
                                                                    vocab['<nohs>'], vocab_full['<nohs>'])}

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

    model_file = 'all_base_TEST'

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/hyps_base_bert_test_2020-05-26-22-10-40.json', 'r') as f:
        hyps42 = json.load(f)

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/hyps_base_bert_test_2020-05-26-22-14-22.json', 'r') as f:
        hyps1 = json.load(f)

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/hyps_base_bert_test_2020-05-26-22-18-4.json', 'r') as f:
        hyps2 = json.load(f)

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/hyps_base_bert_test_2020-05-26-22-22-3.json', 'r') as f:
        hyps3 = json.load(f)

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/hyps_base_bert_test_2020-05-26-22-25-20.json', 'r') as f:
        hyps4 = json.load(f)

    with open('../FINAL_SPEAKERS/TEST_JSONS/base/refs_base_bert_test_2020-05-26-22-10-40.json', 'r') as f:
        refs = json.load(f)

    hyps42_first = []
    hyps1_first = []
    hyps2_first = []
    hyps3_first = []
    hyps4_first = []

    hyps42_later = []
    hyps1_later = []
    hyps2_later = []
    hyps3_later = []
    hyps4_later = []

    refs_first = []
    refs_later = []

    for i, data in enumerate(test_loader):

        prev_hist_actual = data['actual_prev_utterance']

        actual_prev_string = [vocab_full.index2word[w.item()] for w in prev_hist_actual[0]] # single instance batch
        actual_prev_string = ' '.join(actual_prev_string)

        if actual_prev_string == '<nohs>':
            hyps42_first.append(hyps42[i])
            hyps1_first.append(hyps1[i])
            hyps2_first.append(hyps2[i])
            hyps3_first.append(hyps3[i])
            hyps4_first.append(hyps4[i])

            refs_first.append(refs[i])

        else:
            hyps42_later.append(hyps42[i])
            hyps1_later.append(hyps1[i])
            hyps2_later.append(hyps2[i])
            hyps3_later.append(hyps3[i])
            hyps4_later.append(hyps4[i])

            refs_later.append(refs[i])

    output_dict_42 = (hyps42_first, hyps42_later)
    output_dict_1 = (hyps1_first, hyps1_later)
    output_dict_2 = (hyps2_first, hyps2_later)
    output_dict_3 = (hyps3_first, hyps3_later)
    output_dict_4 = (hyps4_first, hyps4_later)

    print(len(refs), len(refs_first), len(refs_later))

    output_dicts = {'42': output_dict_42,
                    '1': output_dict_1,
                    '2': output_dict_2,
                    '3': output_dict_3,
                    '4': output_dict_4,}

    for s in output_dicts:

        output = output_dicts[s]

        print(s)
        print('first')
        metric_dict = nlge.compute_metrics(refs_first, output[0])
        print(metric_dict)

        (P, R, Fs), hashname = score(output[0], refs_first, lang='en', return_hash=True, model_type="bert-base-uncased")
        print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')

        print('later')
        metric_dict = nlge.compute_metrics(refs_later, output[1])
        print(metric_dict)

        (P, R, Fs), hashname = score(output[1], refs_later, lang='en', return_hash=True, model_type="bert-base-uncased")
        print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')



