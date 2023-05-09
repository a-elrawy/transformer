import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Implement the MultiHeadAttention class. Multi-head attention allows the model to jointly attend to
    information from different representation subspaces at different positions. With a single attention head,
    averaging inhibits this.
    :arg d_model: embedding dimension
    :arg nhead: number of heads
    :arg dropout: dropout rate
"""

    def __init__(self, d_model, nhead, dropout=0.1, *args, **kwargs):
        """Initialize the MultiHeadAttention class.
        :arg d_model: embedding dimension
        :arg nhead: number of Attention heads
        :arg dropout: dropout rate"""
        super().__init__(*args, **kwargs)
        self.nhead = nhead
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        # Query, Key, Value weight matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None, dropout=None):
        """Compute the MultiHeadAttention.
        :arg query: query tensor of shape (N, S, E)
        :arg key: key tensor of shape (N, S, E)
        :arg value: value tensor of shape (N, S, E)
        :arg mask: mask tensor of shape (S, S)
        :arg dropout: dropout rate
        :return: tensor of shape (N, S, E)"""

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(q.size(0), q.size(1), self.nhead, self.d_k).transpose(1, 2)  # (N, h, S, d_k)
        k = k.view(k.size(0), k.size(1), self.nhead, self.d_k).transpose(1, 2)  # (N, h, S, d_k)
        v = v.view(v.size(0), v.size(1), self.nhead, self.d_k).transpose(1, 2)  # (N, h, S, d_k)
        # N = batch size, h = number of heads, S = sequence length, d_k = embedding dimension / number of heads

        out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (N, h, S, S)

        if mask is not None:
            out = out.masked_fill(mask == 0, float('-inf'))

        out = self.softmax(out)

        if dropout is not None:
            self.dropout(out)

        out = torch.matmul(out, v)  # (N, h, S, d_k)

        out = out.view(out.size(0), out.size(2), -1)
        out = self.w_o(out)

        return out
