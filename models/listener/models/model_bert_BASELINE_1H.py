import torch
import torch.nn as nn
import torch.nn.functional as F


class ListenerModelBertBaseline1H(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, img_dim, att_dim, dropout_prob):
        super().__init__()
        self.embedding_dim = embedding_dim  # not used
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim  # not used
        self.attention_dim = att_dim  # not used

        # project images to hid dim
        self.linear_separate = nn.Linear(354, self.hidden_dim)  # no of images in the dataset is 354
        self.lin_out = nn.Linear(self.hidden_dim, 1)  # directly convert to scalar

        self.tanh = nn.Tanh()  # not used
        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()  # initialize layers

    def init_weights(self):

        for ll in [self.linear_separate, self.lin_out]:

            ll.bias.data.fill_(0)
            ll.weight.data.uniform_(-0.1, 0.1)

    def forward(self, representations, lengths, separate_images, visual_context, prev_hist, masks, device):

        """
        @param representations: not used in this model
        @param lengths: not used in this model
        @param separate_images: one-hot vectors based on image IDs for all 6 images in the context separately
        @param visual_context: not used in this model
        @param prev_hist: not used in this model
        @param masks: not used in this model
        @param device: not used in this model
        """

        # one-hot vectors per image in context are processed
        separate_images = self.dropout(separate_images)
        separate_images = self.linear_separate(separate_images)
        separate_images = self.relu(separate_images)
        separate_images = F.normalize(separate_images, p=2, dim=2)

        # final vectors are directly mapped to scalars
        dot = self.lin_out(separate_images)

        return dot
