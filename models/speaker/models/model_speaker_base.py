import torch
import torch.nn as nn


class SpeakerModelBase(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, img_dim, dropout_prob):
        super().__init__()
        self.vocab_size = vocab_size - 1  # to exclude <nohs> from the decoder (but add for embed and encoder)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.dropout_prob = dropout_prob

        # embeddings learned from scratch (adding +1 because the embedding for nohs is also learned)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embedding_dim, padding_idx=0, scale_grad_by_freq=True)

        # LSTM decoder for generating the next utterance
        self.lstm_decoder = nn.LSTMCell(self.hidden_dim, self.hidden_dim, bias=True)

        self.linear = nn.Linear(self.img_dim, int(self.hidden_dim))

        self.linear_hid = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        # concatenated visual context is projected to hidden dimensions
        self.lin_viscontext = nn.Linear(self.img_dim*6, self.hidden_dim)

        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)

        # project the output of decoder to the vocabulary size (nohs excluded)
        self.lin2voc = nn.Linear(self.hidden_dim, self.vocab_size)

        self.lin_mm = nn.Linear(self.hidden_dim+self.embedding_dim, self.hidden_dim)

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        for ll in [self.linear, self.linear_separate, self.linear_hid, self.lin2voc, self.lin_viscontext, self.lin_mm]:
            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, utterance, lengths, prev_utterance, prev_utt_lengths, visual_context_sep, visual_context,
                target_img_feats, targets, prev_hist, prev_hist_len, normalize, masks, device):

        """
        @param utterance: ground-truth subsequent utterance converted into indices using the reduced vocabulary,
        which will be fed into the decoder during teacher forcing
        @param lengths: utterance lengths
        @param prev_utterance: not used in this model
        @param prev_utt_lengths: not used in this model
        @param visual_context_sep: image feature vectors for all 6 images in the context separately
        @param visual_context: concatenation of 6 images in the context
        @param target_img_feats: features of the image for which we will generate a new utterance
        @targets, prev_hist, prev_hist_len, normalize, param masks: not used in this model
        @param device: device to which the tensors are moved
        """

        batch_size = utterance.shape[0]  # effective batch size
        decode_length = utterance.shape[1] - 1 # teacher forcing (except eos)

        # this model uses only the visual input to start off the decoder to generate the next utterance
        # therefore, visual context and target image features are processed
        visual_context_hid = self.relu(self.lin_viscontext(self.dropout(visual_context)))
        target_img_hid = self.relu(self.linear_separate(self.dropout(target_img_feats)))

        # word prediction scores
        predictions = torch.zeros(batch_size, decode_length, self.vocab_size).to(device)

        # decoder hidden state is initialised with the concatenation of visual context and target image for which
        # we will generate a new utterance
        decoder_hid = self.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1))

        h1, c1 = decoder_hid, decoder_hid

        # teacher forcing during training, decoder input: ground-truth subsequent utterance
        target_utterance_embeds = self.embedding(utterance)

        for l in range(decode_length):

            # decoder takes the concatenation of the ground-truth token's embedding and decoder's initial hidden state
            h1, c1 = self.lstm_decoder(self.lin_mm(torch.cat((target_utterance_embeds[:,l], decoder_hid),dim=1)), hx=(h1,c1))

            # hidden state of the decoder is projected to vocabulary size to predict the word to be generated
            word_pred = self.lin2voc(h1)

            predictions[:, l] = word_pred

        return predictions