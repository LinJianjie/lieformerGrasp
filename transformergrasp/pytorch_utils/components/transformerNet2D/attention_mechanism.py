import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import torch.nn.functional as F

from pytorch_utils.components.transformerNet2D.utils import *


def attention3D(query, key, value, mask=None, dropout=None, local_attention_size=None, make_future=False):
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
    B, N, D, L = query.shape
    query = query.permute(0, 2, 1, 3).reshape(B * D, N, L)
    key = key.permute(0, 2, 1, 3).reshape(B * D, N, L)
    value = value.permute(0, 2, 1, 3).reshape(B * D, N, L)
    scores = torch.bmm(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    # print("scores: ", scores.shape)
    # Compute local map mask
    if local_attention_size is not None:
        attention_mask = generate_local_square_map_mask(scores.shape[1], scores.shape[2], local_attention_size,
                                                        mask_future=False)
        scores = scores.masked_fill(attention_mask, float('-inf'))

    # Compute future mask
    if mask == "subsequent":
        future_mask = subsequent_mask(query.shape[1]).cuda()
        scores = scores.masked_fill(future_mask == 0, float('-inf'))

    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = F.dropout(p_attention, p=dropout)
    x_attention = torch.bmm(p_attention, value)
    x_attention = x_attention.view(B, D, N, L).permute(0, 2, 1, 3)
    return x_attention, p_attention


class MultiHeadedAttention3D(nn.Module):
    def __init__(self, num_heads, d_model, d_in=None, dropout=0.1, local_attention_size=None):
        super(MultiHeadedAttention3D, self).__init__()
        # assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        if d_in is None:
            d_in = d_model
        self._W_q = nn.Conv2d(d_in, d_model, kernel_size=1)
        self._W_k = nn.Conv2d(d_in, d_model, kernel_size=1)
        self._W_v = nn.Conv2d(d_in, d_model, kernel_size=1)
        # self._W_qkv = nn.Linear(d_model, d_model * 3)
        # Output linear function
        self._W_o = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.attn = None
        self.dropout = dropout
        self.local_attention_size = local_attention_size

    def forward(self, X_Query, X_Key, X_Value, mask=None, offset_attention=False):
        nbatches = X_Query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        # qkv = self._W_qkv(X_Query).chunk(3, dim=1)
        # q, k, v = qkv
        q = self._W_q(X_Query)
        k = self._W_k(X_Key)
        v = self._W_v(X_Value)
        # print("q: ", q.shape)
        queries = torch.cat(q.chunk(self.num_heads, dim=-1), dim=0)
        keys = torch.cat(k.chunk(self.num_heads, dim=-1), dim=0)
        values = torch.cat(v.chunk(self.num_heads, dim=-1), dim=0)
        # print("queries: ", queries.shape)
        x, self.attn = attention3D(queries, keys, values, mask=mask, dropout=self.dropout,
                                   local_attention_size=self.local_attention_size)
        attention_heads = torch.cat(x.chunk(self.num_heads, dim=0), dim=-1)
        # print("attention_heads: ", attention_heads.shape)
        # queries = self._W_q(X_Query).chunk(self.num_heads, dim=-1)
        # keys = self._W_k(X_Key).chunk(self.num_heads, dim=-1)
        # values = self._W_v(X_Value).chunk(self.num_heads, dim=-1)
        # atten_ = []
        # for i in range(self.num_heads):
        #     x, _ = attention(queries[i], keys[i], values[i], mask=None, dropout=self.dropout,
        #                      local_attention_size=None)
        #     atten_.append(x)
        #
        # attention_heads = torch.cat(atten_, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        # offset attention mechanishem
        if offset_attention:
            self_attention = self._W_o(X_Query - attention_heads)
        else:
            self_attention = self._W_o(attention_heads)
        return self_attention


def attention2D(query, key, value, mask=None, dropout=None, local_attention_size=None, make_future=False):
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
        scores = scores.masked_fill(attention_mask, float('-inf'))

    # Compute future mask
    if mask == "subsequent":
        future_mask = subsequent_mask(query.shape[1]).cuda()
        scores = scores.masked_fill(future_mask == 0, float('-inf'))

    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = F.dropout(p_attention, p=dropout)
    x_attention = torch.bmm(p_attention, value)
    return x_attention, p_attention


class MultiHeadedAttention2D(nn.Module):
    def __init__(self, num_heads, d_model, d_in=None, dropout=0.1, local_attention_size=None):
        super(MultiHeadedAttention2D, self).__init__()
        # assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        if d_in is None:
            d_in = d_model
        self._W_q = nn.Conv1d(d_in, d_model, kernel_size=1)
        self._W_k = nn.Conv1d(d_in, d_model, kernel_size=1)
        self._W_v = nn.Conv1d(d_in, d_model, kernel_size=1)
        # self._W_qkv = nn.Linear(d_model, d_model * 3)
        # Output linear function
        self._W_o = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.attn = None
        self.dropout = dropout
        self.local_attention_size = local_attention_size

    def forward(self, X_Query, X_Key, X_Value, mask=None, offset_attention=False):
        nbatches = X_Query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        # qkv = self._W_qkv(X_Query).chunk(3, dim=1)
        # q, k, v = qkv
        q = self._W_q(X_Query)
        k = self._W_k(X_Key)
        v = self._W_v(X_Value)
        queries = torch.cat(q.chunk(self.num_heads, dim=-1), dim=0)
        keys = torch.cat(k.chunk(self.num_heads, dim=-1), dim=0)
        values = torch.cat(v.chunk(self.num_heads, dim=-1), dim=0)
        x, self.attn = attention2D(queries, keys, values, mask=mask, dropout=self.dropout,
                                   local_attention_size=self.local_attention_size)
        attention_heads = torch.cat(x.chunk(self.num_heads, dim=0), dim=-1)
        # queries = self._W_q(X_Query).chunk(self.num_heads, dim=-1)
        # keys = self._W_k(X_Key).chunk(self.num_heads, dim=-1)
        # values = self._W_v(X_Value).chunk(self.num_heads, dim=-1)
        # atten_ = []
        # for i in range(self.num_heads):
        #     x, _ = attention(queries[i], keys[i], values[i], mask=None, dropout=self.dropout,
        #                      local_attention_size=None)
        #     atten_.append(x)
        #
        # attention_heads = torch.cat(atten_, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        # offset attention mechanishem
        if offset_attention:
            self_attention = self._W_o(X_Query - attention_heads)
        else:
            self_attention = self._W_o(attention_heads)
        return self_attention


def attention1D(query, key, value, mask=None, dropout=None, local_attention_size=None, make_future=False):
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
    # d_k = 1
    query_new = query.unsqueeze(-1)
    key_new = key.unsqueeze(-1)
    scores = torch.bmm(query_new, key_new.transpose(-2, -1)) / np.sqrt(d_k)
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
    # if dropout is not None:
    #     p_attention = F.dropout(p_attention, p=dropout)
    value_new = value.unsqueeze(dim=-1)
    x_attention = torch.bmm(p_attention, value_new).squeeze()
    return x_attention, p_attention


class MultiHeadedAttention1D(nn.Module):
    def __init__(self, num_heads, d_model, d_in=None, dropout=0.1, local_attention_size=None):
        super(MultiHeadedAttention1D, self).__init__()
        # assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        if d_in is None:
            d_in = d_model
        self._W_q = nn.Linear(d_in, d_model)
        self._W_k = nn.Linear(d_in, d_model)
        self._W_v = nn.Linear(d_in, d_model)
        # self._W_qkv = nn.Linear(d_model, d_model * 3)
        # Output linear function
        self._W_o = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = dropout
        self.local_attention_size = local_attention_size

    def forward(self, X_Query, X_Key, X_Value, mask=None):
        nbatches = X_Query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        # qkv = self._W_qkv(X_Query).chunk(3, dim=1)
        # q, k, v = qkv
        q = self._W_q(X_Query)
        k = self._W_k(X_Key)
        v = self._W_v(X_Value)
        queries = torch.cat(q.chunk(self.num_heads, dim=-1), dim=0)
        keys = torch.cat(k.chunk(self.num_heads, dim=-1), dim=0)
        values = torch.cat(v.chunk(self.num_heads, dim=-1), dim=0)
        x, self.attn = attention1D(queries, keys, values, mask=mask, dropout=self.dropout,
                                   local_attention_size=self.local_attention_size)
        attention_heads = torch.cat(x.chunk(self.num_heads, dim=0), dim=-1)

        # queries = self._W_q(X_Query).chunk(self.num_heads, dim=-1)
        # keys = self._W_k(X_Key).chunk(self.num_heads, dim=-1)
        # values = self._W_v(X_Value).chunk(self.num_heads, dim=-1)
        # atten_ = []
        # for i in range(self.num_heads):
        #     x, _ = attention(queries[i], keys[i], values[i], mask=None, dropout=self.dropout,
        #                      local_attention_size=None)
        #     atten_.append(x)
        #
        # attention_heads = torch.cat(atten_, dim=2)
        # 3) "Concat" using a view and apply a final linear.
        # offset attention mechanishem
        self_attention = self._W_o(X_Query - attention_heads)
        return self_attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1, local_attention_size=None, use_2D=True):
        super(MultiHeadedAttention, self).__init__()
        if use_2D:
            self.multi = MultiHeadedAttention2D(num_heads=num_heads, d_model=d_model,
                                                d_in=d_model,
                                                dropout=dropout,
                                                local_attention_size=local_attention_size)
        else:
            self.multi = MultiHeadedAttention1D(num_heads=num_heads, d_model=d_model,
                                                d_in=d_model,
                                                dropout=dropout,
                                                local_attention_size=local_attention_size)

    def forward(self, X_Query, X_Key, X_Value, mask=None):
        return self.multi(X_Query, X_Key, X_Value, mask)


class MultiHeadedAttentionXD(nn.Module):
    def __init__(self, num_heads, d_model, d_in=None, dropout=0.1, local_attention_size=None, choose_XD=1):
        super(MultiHeadedAttentionXD, self).__init__()
        if choose_XD == 1:
            self.multi = MultiHeadedAttention1D(num_heads, d_model, d_in, dropout, local_attention_size)
        elif choose_XD == 2:
            self.multi = MultiHeadedAttention2D(num_heads, d_model, d_in, dropout, local_attention_size)
        elif choose_XD == 3:
            self.multi = MultiHeadedAttention3D(num_heads, d_model, d_in, dropout, local_attention_size)
        else:
            raise NotImplementedError("Invalid MultiHeadedAttentionXD mode: {}".format(choose_XD))

    def forward(self, X_Query, X_Key, X_Value, mask=None):
        return self.multi(X_Query, X_Key, X_Value, mask)


if __name__ == '__main__':
    x = torch.rand(2, 67, 64, 1000)
    msa = MultiHeadedAttention3D(num_heads=8, d_model=128, d_in=67)
    msa(x, x, x)
