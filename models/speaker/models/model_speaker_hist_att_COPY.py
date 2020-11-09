import torch
import torch.nn as nn


class SpeakerModelHistAttCopy(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, img_dim, dropout_prob, attention_dim):
        super().__init__()
        self.vocab_size = vocab_size - 1  # to exclude <nohs> from the decoder (but add for embed and encoder)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.dropout_prob = dropout_prob

        # attention over encoder steps
        self.attention_dim = attention_dim

        # embeddings learned from scratch (adding +1 because the embedding for nohs is also learned)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim, padding_idx=0, scale_grad_by_freq=True)

        # Bidirectional LSTM encoder for the previous utterance
        self.lstm_encoder = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1,
                                    batch_first=True, bidirectional=True)  # BIDIRECTIONAL

        # LSTM decoder for generating the next utterance
        self.lstm_decoder = nn.LSTMCell(self.embedding_dim, self.hidden_dim, bias=True)

        self.linear_hid = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.linear_dec = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.lin_viscontext = nn.Linear(self.img_dim*6, self.hidden_dim)

        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        # attention related hidden layers
        self.lin2att_hid = nn.Linear(self.hidden_dim, self.attention_dim)

        self.lin2att_hist = nn.Linear(self.hidden_dim*2, self.attention_dim)  # 2 because of BiLSTM

        # project to vocabulary size
        self.lin_hid2mid = nn.Linear(self.attention_dim + self.hidden_dim, self.hidden_dim)
        self.lin_mid2voc = nn.Linear(self.hidden_dim, self.vocab_size)

        self.lin_mm = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        # p_gen related linear layers
        self.lin2pgen_enccontext = nn.Linear(self.hidden_dim, 1)
        self.lin2pgen_decinp = nn.Linear(self.embedding_dim, 1)
        self.lin2pgen_decstate = nn.Linear(self.hidden_dim, 1)
        self.linear_coverage = nn.Linear(1, self.attention_dim)  # not used

        self.attention = nn.Linear(self.attention_dim, 1)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        for ll in [self.linear_hid, self.linear_dec, self.linear_separate, self.lin_hid2mid, self.lin_mid2voc, self.lin_viscontext, self.lin_mm,
                   self.lin2att_hist, self.lin2att_hid, self.attention,
                   self.lin2pgen_decinp, self.linear_coverage, self.lin2pgen_enccontext, self.lin2pgen_decstate]:

            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, utterance, lengths, prev_utterance, prev_utt_lengths, visual_context_sep, visual_context,
                target_img_feats, targets, prev_hist, prev_hist_len, normalize, masks, device,
                unk_words, mapped_prev_words, coverage_flag, masks_nohs):

        """
        @param utterance: ground-truth subsequent utterance converted into indices using the reduced vocabulary,
        which will be fed into the decoder during teacher forcing
        @param lengths: utterance lengths
        @param prev_utterance: if exists, the previous utterance for the image, if not <nohs>
        @param prev_utt_lengths: length of the previous utterance
        @param visual_context_sep: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param target_img_feats: features of the image for which we will generate a new utterance
        @param targets, prev_hist, prev_hist_len, normalize: not used
        @param masks: masks for pad tokens
        @param device: device to which the tensors are moved
        @param unk_words: list of unknown words in the previous utterance
        @param mapped_prev_words: previous utterance mapped to extended vocabulary
        @param coverage_flag: not used
        @param masks_nohs: if there is no previous utterance, mask nohs
        """

        # encoder input: prev_utterance (previous utterance with UNK tokens), or NOHS

        # decoder input: utterance (sentence to be generated with UNK tokens, teacher-forcing)

        # target generation: mapped utterance for the calculation of loss
        # decoder target with words from utterance input kept the same and the UNKs are replaced with
        # temporary IDs added to the vocabulary (if that word exists in actual previous utterance, otherwise still UNK)

        # reference sentence: actual words of the utterance (to calculate metric scores, we give the original sentences)

        batch_size = utterance.shape[0] # effective batch size
        decode_length = utterance.shape[1] - 1 # teacher forcing (except eos)

        # unks in previous utterance
        len_unk_words_batch = unk_words.shape[1]  # batch, max_unk

        # visual context and target image features are processed
        visual_context_hid = self.relu(self.lin_viscontext(self.dropout(visual_context)))
        target_img_hid = self.relu(self.linear_separate(self.dropout(target_img_feats)))

        # concatenated visual input (context; target)
        concat_visual_input = self.relu(self.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1)))

        # previous utterance is embedded
        embeds_words = self.dropout(self.embedding(prev_utterance))  # b, l, d

        # pack sequence
        sorted_prev_utt_lens, sorted_idx = torch.sort(prev_utt_lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        concat_visual_input = concat_visual_input[sorted_idx]

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds_words, sorted_prev_utt_lens, batch_first=True)

        # start LSTM encoder conditioned on the visual input
        concat_visual_input = torch.stack((concat_visual_input, concat_visual_input), dim=0)

        # feed the previous utterance into the LSTM encoder
        packed_outputs, hidden = self.lstm_encoder(packed_input, hx=(concat_visual_input, concat_visual_input))

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # already concat forward backward (timestep t, same position)

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        batch_out_hidden = hidden[0][:, reversed_idx]

        # teacher forcing

        # extend vocabulary size with unknown words
        extended_vocab_size = self.vocab_size + len_unk_words_batch

        # word prediction scores
        predictions = torch.zeros(batch_size, decode_length, extended_vocab_size).to(device)

        if len_unk_words_batch == 0:
            len_unk_words_batch += 1
            nohs_flag = True

            # if there are no unknown tokens in the whole batch, there could be issues in the further code
            # when we have NOHS from the encoder
            # as it maps to the very last index in the vocabulary, where we now only add temporary IDs of unknowns
            # we still pay attention to NOHS and thus,
            # scatter would want to add its attention to the extended vocabulary

            # therefore, we add a dummy index at the end to account for NOHS in scatter
            # then mask and discard it at the end

            # if there is a proper previous utterance, it will still add one index as well but that will contain 0
            # because there would be no word that maps to the last index in the vocabulary

        else:
            nohs_flag = False

            # there are unknown words in the batch, which will be properly mapped
            # attention to nohs will be masked, so they don't contribute to the first temporary ID incorrectly

        # forward backward concatenation of encoder's last hidden states
        decoder_hid = self.linear_dec(torch.cat((batch_out_hidden[0], batch_out_hidden[1]), dim=1))

        history_att = self.lin2att_hist(outputs)

        # start decoder with the hidden states of the encoder
        h1, c1 = decoder_hid, decoder_hid

        # teacher forcing during training, decoder input: ground-truth subsequent utterance
        target_utterance_embeds = self.embedding(utterance)
        # this is the input to the decoder with the unk vectors
        # but what is expected at the output level would include the original words
        # in ref and temporary unk IDs in loss

        # decoder input for p_gen
        decoder_input_gen = self.tanh(self.lin2pgen_decinp(target_utterance_embeds))

        covloss = torch.zeros(batch_size).to(device)

        coverage_vector = torch.zeros(batch_size, prev_utterance.shape[1], 1).to(device) # coverage is all 0 at the beginning

        for l in range(decode_length):

            # decoder takes target word embeddings
            h1, c1 = self.lstm_decoder(target_utterance_embeds[:,l], hx=(h1,c1))

            # use h1 in attention calculations over the history
            h1_att = self.lin2att_hid(h1)

            # attention calculation (previous utterance and current decoder state interacts)

            if coverage_flag:

                attention_out = self.attention(self.tanh(history_att + h1_att.unsqueeze(1)
                                                         + self.linear_coverage(coverage_vector)))

            else:

                attention_out = self.attention(self.tanh(history_att + h1_att.unsqueeze(1)))

            # pad tokens in the previous utterance to mask them out
            attention_out = attention_out.masked_fill_(masks, float('-inf'))

            # final attention weights
            att_weights = self.softmax(attention_out)

            # encoder context representation
            att_context_vector = (history_att * att_weights).sum(dim=1)

            # encoder context for p_gen
            att_context_vector_pgen = self.tanh(self.lin2pgen_enccontext(att_context_vector))

            # decoder state for p_gen
            h1_gen = self.tanh(self.lin2pgen_decstate(h1))

            # calculate p_gen
            p_gen = torch.sigmoid(att_context_vector_pgen + h1_gen + decoder_input_gen[:, l])  # soft switch

            # calculate logprobabilities of generating a word from the vocabulary
            p_vocab = self.logsoftmax(self.lin_mid2voc(self.lin_hid2mid(torch.cat((h1, att_context_vector), dim=1))))

            # extend the vocab to add the temporary unk IDs (for each example, the same temporary ID can refer to different words)
            p_vocab_extended = torch.cat((p_vocab, torch.zeros(batch_size, len_unk_words_batch).to(device)), dim=1)

            final_gen_distribution = p_gen * p_vocab_extended  # generation probability for words actually in the vocabulary

            # pay attention to nohs, but don't add its attention to vocabulary probabilities
            # (because we don't want to generate it)

            nohs_masked_att_weights = att_weights.clone().masked_fill_(masks_nohs, 0.0)

            # attention over words in mapped previous utterance
            final_att_distribution = (1 - p_gen) * self.logsoftmax(nohs_masked_att_weights.squeeze(2))

            # scatter attention to extended vocabulary (attention of words in mapped previous utterance)

            total_word_prob = final_gen_distribution.scatter_add(1, mapped_prev_words, final_att_distribution)

            if nohs_flag:
                # no unknown added, but we had NOHS to account for, discard it
                total_word_prob = total_word_prob[:, :-1]

            if coverage_flag:
                # not used
                cur_covloss = torch.min(att_weights, coverage_vector).sum(dim=1)

                covloss += cur_covloss.squeeze()

                coverage_vector = coverage_vector + att_weights

            predictions[:, l] = total_word_prob

        return predictions, covloss  # covloss not used