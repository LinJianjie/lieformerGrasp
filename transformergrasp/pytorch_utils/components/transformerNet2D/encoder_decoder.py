import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_utils.components.netUtils import clones
from pytorch_utils.components.transformerNet2D.attention_mechanism import MultiHeadedAttention, MultiHeadedAttentionXD
from pytorch_utils.components.transformerNet2D.utils import PointNetEmbeddings, PositionFeedForward, \
    PositionFeedForwardXD
from pytorch_utils.components.netUtils import NetUtil


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, layer_norm_size, dropout, use_2D=True):
        super(ResidualConnection, self).__init__()
        if use_2D:
            self.norm = nn.BatchNorm1d(layer_norm_size)
        else:
            self.norm = nn.LayerNorm(layer_norm_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + F.relu(self.norm((self.dropout(sublayer(x)))))
        # return x + F.relu(self.norm((sublayer(x))))


class ResidualConnectionXD(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, layer_norm_size, dropout, choose_XD=1):
        super(ResidualConnectionXD, self).__init__()
        if choose_XD == 2:
            self.norm = nn.BatchNorm1d(layer_norm_size)
        elif choose_XD == 1:
            self.norm = nn.LayerNorm(layer_norm_size)
        elif choose_XD == 3:
            self.norm = nn.BatchNorm2d(layer_norm_size)
        else:
            raise NotImplementedError("Invalid BatchNormalization mode: {}".format(choose_XD))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + F.relu(self.norm((self.dropout(sublayer(x)))))


class PointEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff=1024, num_heads=8, dropout=0.1, local_attention_size=None,
                 make_future=False, use_2D=True):
        super(PointEncoderLayer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_attention_size = local_attention_size
        self.make_future = make_future
        self.self_attention = MultiHeadedAttention(num_heads, d_model, dropout=self.dropout,
                                                   local_attention_size=None, use_2D=use_2D)
        self.feed_forward = PositionFeedForward(d_model=d_model, d_ff=d_ff, use_2D=use_2D)

        self.sublayer = clones(ResidualConnection(d_model, dropout, use_2D=use_2D), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class PointEncoderLayerXD(nn.Module):
    def __init__(self, d_model, d_in=None, d_ff=1024, num_heads=8, dropout=0.1, local_attention_size=None,
                 make_future=False, choose_XD=1):
        super(PointEncoderLayerXD, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_attention_size = local_attention_size
        self.make_future = make_future
        self.self_attention = MultiHeadedAttentionXD(num_heads, d_model, d_in=d_in, dropout=self.dropout,
                                                     local_attention_size=None, choose_XD=choose_XD)
        self.feed_forward = PositionFeedForwardXD(d_model=d_model, d_ff=d_ff, choose_XD=choose_XD)

        self.sublayer = clones(ResidualConnectionXD(d_model, dropout, choose_XD=choose_XD), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class PointEncoder(nn.Module):
    def __init__(self, encode_layer, N, input_linear_transform=None, use_cat=True, use_2D=True):
        super(PointEncoder, self).__init__()
        self.input_linear_transform = input_linear_transform
        self.layers = clones(encode_layer, N)
        self.use_cat = use_cat
        self.N = N
        if use_cat:
            self.last_layer_size = encode_layer.d_model * self.N
        else:
            self.last_layer_size = encode_layer.d_model
        if use_2D:
            self.batch_norm = nn.BatchNorm1d(self.last_layer_size)
        else:
            self.batch_norm = nn.LayerNorm(self.last_layer_size)

    def forward(self, x):
        if self.input_linear_transform is not None:
            x = self.input_linear_transform(x)
        hidden_state = []
        for layer in self.layers:
            x = layer(x)
            hidden_state.append(x)
        if self.use_cat:
            x = torch.cat(hidden_state, dim=1)
        return self.batch_norm(x)


class PointEncoderXD(nn.Module):
    def __init__(self, encode_layer, N, input_linear_transform=None, use_cat=True, choose_XD=1):
        super(PointEncoderXD, self).__init__()
        self.input_linear_transform = input_linear_transform
        self.layers = clones(encode_layer, N)
        self.use_cat = use_cat
        self.N = N
        if use_cat:
            self.last_layer_size = encode_layer.d_model * self.N
        else:
            self.last_layer_size = encode_layer.d_model
        if choose_XD == 2:
            self.batch_norm = nn.BatchNorm1d(self.last_layer_size)
        elif choose_XD == 1:
            self.batch_norm = nn.LayerNorm(self.last_layer_size)
        elif choose_XD == 3:
            self.batch_norm = nn.BatchNorm2d(self.last_layer_size)
        else:
            raise NotImplementedError("Invalid bathNormalize: {}".format(choose_XD))

    def forward(self, x):
        if self.input_linear_transform is not None:
            x = self.input_linear_transform(x)
        hidden_state = []
        for layer in self.layers:
            x = layer(x)
            hidden_state.append(x)
        if self.use_cat:
            x = torch.cat(hidden_state, dim=1)
        return self.batch_norm(x)


class TransformerEncoder(nn.Module):
    def __init__(self, N=6, d_model=512, d_ff=1024, num_head=8, dropout=0.1, use_cMLP=False, use_2D=True,
                 pointNet=None):
        super(TransformerEncoder, self).__init__()
        self.num_encoder = N
        self.encoder = PointEncoder(
            encode_layer=PointEncoderLayer(d_model=d_model, d_ff=d_ff,
                                           num_heads=num_head, dropout=dropout,
                                           local_attention_size=None, use_2D=use_2D),
            input_linear_transform=PointNetEmbeddings(d_model=d_model, use_cMLP=use_cMLP, pointNet=pointNet,
                                                      use_2D=use_2D),
            N=self.num_encoder)
        self.last_layer_size = self.encoder.last_layer_size
        if use_2D:
            self.pn_layers = NetUtil.SeqPointNetConv1d(
                channels=[self.encoder.last_layer_size, self.encoder.last_layer_size],
                active="relu")
        else:
            self.pn_layers = NetUtil.SeqLinear(
                channels=[self.encoder.last_layer_size, self.encoder.last_layer_size],
                activation="LeakyReLU")

    def forward(self, src_):
        B, N, D = src_.shape
        z_hat = self.encoder(src_)
        z_hat = self.pn_layers(z_hat)
        return z_hat
