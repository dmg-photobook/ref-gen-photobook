import os
import json
import pickle
import torch
from collections import defaultdict
from torch.utils.data import Dataset

import numpy as np

class SpeakerDataset(Dataset):
    def __init__(self, split, data_dir, chain_file, utterances_file, orig_ref_file, vocab_obj, actual_vocab_obj,
                 vectors_file, subset_size):

        self.data_dir = data_dir
        self.split = split

        self.vocab_full = actual_vocab_obj

        self.vocab = vocab_obj
        # this vocab includes <nohs> for encoding
        # but the decoder predicts over a range of len(vocab)-1
        # here we don't want to copy <nohs>
        # so it will not be generated
        self.vocab_len_decoder = len(self.vocab) - 1
        # unk words will be added to the end of effective decoder vocab

        self.max_len = 0

        # Load a PhotoBook utterance chain dataset
        with open(os.path.join(self.data_dir, chain_file), 'r') as file:
            self.chains = json.load(file)

        # Load an underlying PhotoBook dialogue utterance dataset
        with open(os.path.join(self.data_dir, utterances_file), 'rb') as file:
            self.utterances = pickle.load(file)

        # Original reference sentences without unks
        with open(os.path.join(self.data_dir, orig_ref_file), 'rb') as file:
            self.text_refs = pickle.load(file)

        split_utt_file = utterances_file.split('_')
        utterances_actual_file = split_utt_file[0] + '_COPY_' + split_utt_file[1] + '_' + split_utt_file[2]

        # Load an underlying PhotoBook dialogue utterance dataset
        with open(os.path.join(self.data_dir, utterances_actual_file), 'rb') as file:
            self.utterances_actual = pickle.load(file)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

        self.img_dim = 2048
        self.img_count = 6  # images in the context

        self.data = dict()

        self.img2chain = defaultdict(dict)

        for chain in self.chains:

            self.img2chain[chain['target']][chain['game_id']] = chain['utterances']

        if subset_size == -1:
            self.subset_size = len(self.chains)
        else:
            self.subset_size = subset_size

        print('processing',self.split)

        # every utterance in every chain, along with the relevant history
        for chain in self.chains[:self.subset_size]:

            chain_utterances = chain['utterances']
            game_id = chain['game_id']

            for s in range(len(chain_utterances)):

                # this is the expected target generation
                utterance_id = tuple(chain_utterances[s])  # utterance_id = (game_id, round_nr, messsage_nr, img_id)
                round_nr = utterance_id[1]
                message_nr = utterance_id[2]

                # prev utterance in the chain
                for cu in range(len(chain['utterances'])):

                    if chain['utterances'][cu] == list(utterance_id):
                        if cu == 0:
                            previous_utterance = []
                            previous_utterance_actual = []
                        else:
                            prev_id = chain['utterances'][cu - 1]
                            previous_utterance = self.utterances[tuple(prev_id)]['utterance']
                            previous_utterance_actual = self.utterances_actual[tuple(prev_id)]['utterance']

                        break

                # linguistic histories for images in the context
                # HISTORY before the expected generation (could be after the encoded history)
                prev_chains = defaultdict(list)
                prev_chains_actual = defaultdict(list)
                prev_lengths = defaultdict(int)

                cur_utterance_obj = self.utterances[utterance_id]
                cur_utterance_text_ids= cur_utterance_obj['utterance']

                cur_utterance_obj_actual = self.utterances_actual[utterance_id]
                cur_utterance_text_ids_actual = cur_utterance_obj_actual['utterance']

                length = cur_utterance_obj['length']

                if length > self.max_len:
                    self.max_len = length

                assert len(cur_utterance_text_ids) != 2
                # already had added sos eos into length and IDS version

                unk_vocab_seg = []
                mapped_prev_words = []

                if len(previous_utterance) > 0:

                    # there is a previous utterance, check for its UNK words

                    for uw in range(len(previous_utterance)):

                        # THE IDS OF NON-UNK WORDS ARE DIFFERENT IN ACTUAL VS. UNKED CHAIN
                        # WE KEEP THE VALUES IN THE HISTORY
                        # FOR THE UNKS, WE ADD THEM AS ADDITIONS TO THE VOCAB

                        # WE NEVER USE THE IDS of WORDS IN THE ACTUAL HISTORY!!
                        # AS THOSE COME FROM A VOCAB OF THE COMBINATION OF ALL SPLITS

                        if previous_utterance[uw] == 1:  # UNK IS ALWAYS 1 (PAD IS 0) and so on
                            # add the id of actual word
                            # corresponding to this unk

                            orig_unk_word = previous_utterance_actual[uw]

                            if orig_unk_word in unk_vocab_seg:

                                # don't add the word, add its index

                                idx = unk_vocab_seg.index(orig_unk_word)
                                mapped_prev_words.append(self.vocab_len_decoder + idx)

                            else:
                                # add the word and index
                                unk_vocab_seg.append(orig_unk_word)
                                idx = len(unk_vocab_seg)
                                mapped_prev_words.append(self.vocab_len_decoder + idx - 1)  # len already + 1 because of len voc

                        else:
                            # non-unk word
                            mapped_prev_words.append(previous_utterance[uw])

                mapped_utterance_words = []  # expected generation (next utterance)
                # these are mapped using the train vocab + unks from history of the segment

                if len(unk_vocab_seg) > 0:

                    # there were UNK words in the previous utterance

                    for sw in range(len(cur_utterance_text_ids)):

                        # THE IDS OF NON-UNK WORDS ARE DIFFERENT IN ACTUAL VS. UNKED SEGMENT
                        # WE KEEP THE VALUES IN THE SEGMENT
                        # FOR THE UNKS, WE ADD IF THEY EXIST IN THE HISTORY
                        # IF NOT LEAVE THEM 1

                        # WE NEVER USE THE IDS of WORDS IN THE ACTUAL SEGMENT!!
                        # AS THOSE COME FROM A VOCAB OF THE COMBINATION OF ALL SPLITS

                        if cur_utterance_text_ids[sw] == 1:  # unk

                            orig_unk_word = cur_utterance_text_ids_actual[sw]

                            if orig_unk_word in unk_vocab_seg:
                                mapped_unk_word_id = unk_vocab_seg.index(orig_unk_word)
                                mapped_utterance_words.append(self.vocab_len_decoder + mapped_unk_word_id)

                            else:
                                # leave it as unk (not in vocab and not in prev utt!)
                                mapped_utterance_words.append(1)

                        else:

                            mapped_utterance_words.append(cur_utterance_text_ids[sw])

                else:

                    # no unk words in prev or no prev at all
                    mapped_utterance_words = cur_utterance_text_ids # doesn't change
                        
                images = cur_utterance_obj['image_set']
                target = cur_utterance_obj['target']  # index of correct img

                target_image = images[target[0]]

                images = list(np.random.permutation(images))
                target = [images.index(target_image)]

                context_separate = torch.zeros(self.img_count, self.img_dim)

                im_counter = 0

                reference_chain = []

                for im in images:

                    context_separate[im_counter] = torch.tensor(self.image_features[im])

                    if im == images[target[0]]:
                        target_img_feats = context_separate[im_counter]
                        ref_chain = self.img2chain[im][game_id]

                        for rc in ref_chain:
                            rc_tuple = (rc[0], rc[1], rc[2], im)
                            reference_chain.append(' '.join(self.text_refs[rc_tuple]['utterance']))

                    im_counter += 1

                    if game_id in self.img2chain[im]:  # was there a linguistic chain for this image in this game
                        temp_chain = self.img2chain[im][game_id]

                        hist_utterances = []

                        for t in range(len(temp_chain)):

                            _, t_round, t_message, _ = temp_chain[t] #(game_id, round_nr, messsage_nr, img_id)

                            if t_round < round_nr:
                                hist_utterances.append((game_id, t_round, t_message))

                            elif t_round == round_nr:

                                if t_message < message_nr:
                                    hist_utterances.append((game_id, t_round, t_message))

                        if len(hist_utterances) > 0:

                            # ONLY THE MOST RECENT history
                            for hu in [hist_utterances[-1]]:
                                hu_tuple = (hu[0], hu[1], hu[2], im)
                                prev_chains[im].extend(self.utterances[hu_tuple]['utterance'])
                                prev_chains_actual[im].extend(self.utterances_actual[hu_tuple]['utterance'])

                        else:
                            # no prev reference to that image
                            prev_chains[im] = []
                            prev_chains_actual[im] = []

                    else:
                        # image is in the game but never referred to
                        prev_chains[im] = []
                        prev_chains_actual[im] = []

                    prev_lengths[im] = len(prev_chains[im]) # history for images not prev utterance length

                # ALWAYS 6 IMAGES IN THE CONTEXT

                context_concat = context_separate.reshape(self.img_count * self.img_dim)

                self.data[len(self.data)] = {'utterance': cur_utterance_text_ids,
                                             'actual_utterance': cur_utterance_text_ids_actual,
                                             'image_set': images,
                                             'concat_context': context_concat,
                                             'separate_images': context_separate,
                                             'prev_utterance': previous_utterance,
                                             'actual_prev_utterance': previous_utterance_actual,
                                             'prev_length': len(previous_utterance),
                                             'target':target,
                                             'target_img_feats': target_img_feats,
                                             'length': length,
                                             'prev_histories': prev_chains,
                                             'actual_prev_histories': prev_chains_actual,
                                             'prev_history_lengths': prev_lengths,
                                             'unk_words': unk_vocab_seg,
                                             'mapped_utterance_words': mapped_utterance_words,
                                             'mapped_prev_words': mapped_prev_words,
                                             'reference_chain': reference_chain
                                             }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_collate_fn(device, SOS, EOS, NOHS, NOHS_FULL):

        def collate_fn(data):

            max_utt_length = max(d['length'] for d in data)
            max_prevutt_length = max([d['prev_length'] for d in data])

            max_unk_words = max([len(d['unk_words']) for d in data])

            batch = defaultdict(list)

            for sample in data:

                for key in data[0].keys():

                    if key == 'utterance' or key == 'actual_utterance' or key == 'mapped_utterance_words':

                        padded = sample[key] + [0] * (max_utt_length - sample['length'])

                        # print('utt', padded)

                    elif key == 'prev_utterance' or key == 'actual_prev_utterance' or key == 'mapped_prev_words':

                        if len(sample[key]) == 0:
                            # OTHERWISE pack_padded wouldn't work

                            if key == 'prev_utterance' or key == 'mapped_prev_words':
                                padded = [NOHS] + [0] * (max_prevutt_length - 1) # SPECIAL TOKEN FOR NO HIST
                                # WILL BE MASKED IN MAPPED PREV WORDS, WHEN WE ARE ADDING THE ATTENTION TO VOCAB DIST

                            elif key == 'actual_prev_utterance':
                                padded = [NOHS_FULL] + [0] * (max_prevutt_length - 1) # SPECIAL TOKEN FOR NO HIST

                        else:
                            padded = sample[key] + [0] * (max_prevutt_length - len(sample[key]))

                        # print('prevutt', padded)

                    elif key == 'prev_length':

                        # history for images not prev utterance

                        if sample[key] == 0:
                            # wouldn't work in pack_padded
                            padded = 1

                        else:
                            padded = sample[key]

                    elif key == 'image_set':

                        padded = [int(img) for img in sample['image_set']]

                        # print('img', padded)

                    elif key == 'prev_histories' or key == 'actual_prev_histories':

                        padded = sample[key]

                    elif key == 'unk_words':

                        padded = sample[key] + [0] * (max_unk_words - len(sample[key]))

                    else:
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                # print(key)

                if key in ['separate_images', 'concat_context', 'target_img_feats']:
                    batch[key] = torch.stack(batch[key]).to(device)

                elif key in ['utterance', 'prev_utterance', 'actual_utterance', 'actual_prev_utterance',
                             'target', 'length', 'prev_length', 'unk_words', 'mapped_prev_words', 'mapped_utterance_words']:
                    batch[key] = torch.Tensor(batch[key]).long().to(device)

                    # for instance targets can be long and sent to device immediately

            return batch

        return collate_fn

