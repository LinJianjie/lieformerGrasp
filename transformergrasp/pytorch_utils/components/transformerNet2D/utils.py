import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import math
from pytorch_utils.components.netUtils import NetUtil, CPointNet, PointNet


class PositionFeedForward3D(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionFeedForward3D, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.b_1 = nn.LayerNorm([d_ff])
        # self.w_1_1 = nn.Linear(d_ff, d_ff)
        # self.b_1_1 = nn.LayerNorm([d_ff])
        self.f_1 = nn.Sequential(nn.Conv2d(d_model, d_ff, kernel_size=1), nn.BatchNorm2d(d_ff), nn.ReLU())
        self.w_2 = nn.Conv2d(d_ff, d_model, kernel_size=1)

    def forward(self, x):
        x_1 = self.f_1(x)
        x_2 = self.w_2(x_1)
        return x_2


class PositionFeedForward2D(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionFeedForward2D, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.b_1 = nn.LayerNorm([d_ff])
        # self.w_1_1 = nn.Linear(d_ff, d_ff)
        # self.b_1_1 = nn.LayerNorm([d_ff])
        self.f_1 = nn.Sequential(nn.Conv1d(d_model, d_ff, kernel_size=1), nn.BatchNorm1d(d_ff), nn.ReLU())
        self.w_2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

    def forward(self, x):
        x_1 = self.f_1(x)
        x_2 = self.w_2(x_1)
        return x_2


class PositionFeedForward1D(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionFeedForward1D, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.b_1 = nn.LayerNorm([d_ff])
        # self.w_1_1 = nn.Linear(d_ff, d_ff)
        # self.b_1_1 = nn.LayerNorm([d_ff])
        self.f_1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.LayerNorm(d_ff), nn.ReLU())
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.b_1(self.w_1(x)))))
        x_1 = self.f_1(x)
        x_2 = self.w_2(x_1)
        return x_2


class PositionFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, use_2D=True):
        super(PositionFeedForward, self).__init__()
        if use_2D:
            self.ffn = PositionFeedForward2D(d_model, d_ff, dropout)
        else:
            self.ffn = PositionFeedForward1D(d_model, d_ff, dropout)

    def forward(self, x):
        return self.ffn(x)


class PositionFeedForwardXD(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, choose_XD=1):
        super(PositionFeedForwardXD, self).__init__()
        if choose_XD == 1:
            self.ffn = PositionFeedForward1D(d_model, d_ff, dropout)
        elif choose_XD == 2:
            self.ffn = PositionFeedForward2D(d_model, d_ff, dropout)
        elif choose_XD == 3:
            self.ffn = PositionFeedForward3D(d_model, d_ff, dropout)
        else:
            raise NotImplementedError("Invalid  PositionFeedForwardXD mode: {}".format(choose_XD))

    def forward(self, x):
        return self.ffn(x)


class PointNetEmbeddings(nn.Module):
    def __init__(self, d_model, use_cMLP=False, activation="relu", pointNet=None, use_2D=True):
        super(PointNetEmbeddings, self).__init__()
        # self.lut = nn.Embedding(vocab, d_model)
        # self.d_model = d_model
        # self.linear = nn.Conv1d(vocab, d_model, kernel_size=1)
        # self.linear = nn.Linear(vocab, d_model)
        self.use_2D = use_2D
        if use_cMLP:
            if pointNet is None:
                self.pn = CPointNet([3, 64, 64])
            else:
                self.pn = PointNet
        else:
            print("---> use pointNet in Embedding")
            if pointNet is None:
                print("---> use local pointNet in Embedding")
                channels = [3, 64, 64, 128, d_model]
                self.use_local = True
                self.pn = NetUtil.SeqPointNetConv1d(channels=channels, active=activation)
            else:
                self.pn = pointNet
                self.use_local = False

    def forward(self, x):
        # return self.lut(x) * math.sqrt(self.d_model)
        # instead using embeding, we use linear transformation maps the value to a high dimensional
        B, N, D = x.shape
        if self.use_local:
            x = x.permute(0, 2, 1)
        x = self.pn(x)
        # x = torch.max(x, -1)[0]
        # x = x.view(B, 1, -1)
        # return self.linear(x)
        if self.use_local:
            if self.use_2D:
                return x
            else:
                x = torch.max(x, -1)[0]
                return x
        else:
            if self.use_2D:
                x = x.permute(0, 2, 1)
                return x
            else:
                x = torch.max(x, -1)[0]
                return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    return torch.tril(torch.ones(1, size, size))


def generate_local_square_map_mask(chunk_size, chunk_size_2=None, attention_size=None, mask_future=False):
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    if chunk_size_2 is not None:
        local_map = np.empty((chunk_size, chunk_size_2))
    else:
        local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map)
