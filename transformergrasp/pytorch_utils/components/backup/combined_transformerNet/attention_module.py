import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from components.combined_transformerNet.transformer_utils import *


def attention(query, key, value, mask=None, dropout=None, local_attention_size=None, make_future=False):
    """Compute 'Scaled Dot Product Attention
    where K indicates  the number of sequence
    :param make_future:
    :param local_attention_size: make an attention size
    :param query: Input tensor with shape (batch_size, K, d_model) used to compute queries.
    :param key:  Input tensor with shape (batch_size, K, d_model) used to compute keys.
    :param value: Input tensor with shape (batch_size, K, d_model) used to compute values.
    :param mask: Mask to apply on scores before computing attention. One of ``'subsequent'``, None. Default is None.
    :param dropout:
    """

    d_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    # Compute local map mask
    if local_attention_size is not None:
        attention_mask = generate_local_square_map_mask(scores.shape[1], scores.shape[2], local_attention_size,
                                                        mask_future=False)
        # print(attention_mask)
        scores = scores.masked_fill(attention_mask, float('-inf'))

    # Compute future mask
    if mask == "subsequent":
        future_mask = subsequent_mask(query.shape[1]).cuda()
        scores = scores.masked_fill(future_mask == 0, float('-inf'))

    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = F.dropout(p_attention, p=dropout)
    return torch.bmm(p_attention, value), p_attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, local_attention_size=None):
        super(MultiHeadedAttention, self).__init__()
        # print("d_model: ", d_model, "num_heads: ", num_heads)

        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self._W_q = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self._W_k = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self._W_v = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        # Output linear function
        self._W_o = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.attn = None
        self.dropout = dropout
        self.local_attention_size = local_attention_size

    def forward(self, X_Query, X_Key, X_Value, mask=None):
        nbatches = X_Query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [linearTransform(X).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        #                     for linearTransform, X in zip(self.linears, (X_Query, X_Key, X_Value))]
        # print("query: ", query.shape)
        # print("key: ", key.shape)
        # print("value: ", value.shape)
        # B* T* num_heads*d_model/num_heads
        queries = torch.cat(self._W_q(X_Query).chunk(self.num_heads, dim=-1), dim=0)
        keys = torch.cat(self._W_k(X_Key).chunk(self.num_heads, dim=-1), dim=0)
        values = torch.cat(self._W_v(X_Value).chunk(self.num_heads, dim=-1), dim=0)
        x, self.attn = attention(queries, keys, values, mask=mask, dropout=self.dropout,
                                 local_attention_size=self.local_attention_size)
        # 3) "Concat" using a view and apply a final linear.
        attention_heads = torch.cat(x.chunk(self.num_heads, dim=0), dim=-1)
        # offset attention mechanishem
        self_attention = self._W_o(X_Query - attention_heads)
        return self_attention
