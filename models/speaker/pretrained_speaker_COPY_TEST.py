import torch
import numpy as np

from models.model_speaker_hist_att_COPY import SpeakerModelHistAttCopy

from evals_COPY import eval_beam_histatt_copy

from utils.SpeakerDatasetCopy import SpeakerDataset
from utils.Vocab import Vocab

from nlgeval import NLGEval

import os

import datetime


def mask_attn(actual_num_tokens, max_num_tokens, device):

    masks = []

    for n in range(len(actual_num_tokens)):

        # items to be masked are TRUE
        mask = [False] * actual_num_tokens[n] + [True] * (max_num_tokens - actual_num_tokens[n])

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


def mask_nohs(actual_num_tokens, max_num_tokens, device):

    # mask nohs so that attention scores do not contribute to vocab
    # but still pay attention to it

    masks = []

    for n in range(len(actual_num_tokens)):
        # items to be masked are TRUE

        if actual_num_tokens[n] == 1:
            mask = [True] * max_num_tokens  # NOHS is the only token here, we can have 1 T rest F as well, doesn't matter
        else:
            mask = [False] * max_num_tokens

        masks.append(mask)

    masks = torch.tensor(masks).unsqueeze(2).to(device)

    return masks


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    nlge = NLGEval(no_skipthoughts=True, no_glove=True)

    speaker_files = ['saved_models/model_speaker_COPY_copy_42_bert_2020-05-20-22-17-40.pkl',
                     'saved_models/model_speaker_COPY_copy_1_bert_2020-05-25-20-3-58.pkl',
                     'saved_models/model_speaker_COPY_copy_2_bert_2020-05-25-20-4-39.pkl',
                     'saved_models/model_speaker_COPY_copy_3_bert_2020-05-25-20-5-40.pkl',
                     'saved_models/model_speaker_COPY_copy_4_bert_2020-05-25-20-6-40.pkl']

    for speaker_file in speaker_files:

        seed = 28

        # for reproducibility
        print(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print(speaker_file)

        checkpoint = torch.load(speaker_file, map_location=device)

        args = checkpoint['args']

        model_type = args.model_type

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

        max_len = 30  # for beam search

        img_dim = 2048

        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        att_dim = args.attention_dim

        dropout_prob = args.dropout_prob
        beam_size = args.beam_size

        metric = args.metric

        shuffle = args.shuffle
        normalize = args.normalize
        breaking = args.breaking

        print_gen = args.print

        # depending on the selected model type, we will have a different architecture

        if model_type == 'copy':  # copy

            model = SpeakerModelHistAttCopy(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim).to(device)
            coverage_flag = False

        elif model_type == 'copy_cov': # copy and coverage

            model = SpeakerModelHistAttCopy(len(vocab), embedding_dim, hidden_dim, img_dim, dropout_prob, att_dim).to(device)
            coverage_flag = True

        cov_weight = args.coverage_weight

        batch_size = 1

        load_params_test = {'batch_size': 1, 'shuffle': False,
                            'collate_fn': SpeakerDataset.get_collate_fn(device, vocab['<sos>'], vocab['<eos>'],
                                                                        vocab['<nohs>'], vocab_full['<nohs>'])}
        test_loader = torch.utils.data.DataLoader(testset, **load_params_test)

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        with torch.no_grad():
            model.eval()

            isValidation = False
            isTest = True
            print('\nTest Eval')

            # THIS IS test EVAL_BEAM
            print('beam')

            # best_score and timestamp not so necessary here
            best_score = checkpoint['accuracy']  # cider or bert
            t = datetime.datetime.now()
            timestamp = str(t.date()) + '-' + str(t.hour) + '-' + str(t.minute) + '-' + str(t.second)

            if model_type == 'copy' or model_type == 'copy_cov':
                eval_beam_histatt_copy(test_loader, model, args, best_score, print_gen, device,
                                           beam_size, max_len, vocab, mask_attn, nlge, isValidation, timestamp, vocab_full,
                                           mask_nohs, isTest)
