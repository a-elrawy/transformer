import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Implement the PositionalEncoding class. Positional encoding is a method to add information about
    the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Initialize the PositionalEncoding class.
        :arg d_model: embedding dimension
        :arg dropout: dropout rate
        :arg max_len: maximum length of the sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to the input tensor.
        :arg x: input tensor of shape (S, N, E)
        :return: tensor of shape (S, N, E)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """Implement the FeedForward class. Two linear transformations with a ReLU activation in between.
    """

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, *args, **kwargs):
        """Initialize the FeedForward class.
        :arg d_model: embedding dimension
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate
        """
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Compute the FeedForward output.
        :arg x: input tensor of shape (S, N, E)
        :return: tensor of shape (S, N, E)
        """
        x = self.linear2(self.relu(self.linear1(x)))
        x = self.dropout(x)
        return x


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


class EncoderLayer(nn.Module):
    """Implement the EncoderLayer class. The encoder layer is made up of self-attn and feedforward network."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, *args, **kwargs):
        """Initialize the EncoderLayer class.
        :arg d_model: embedding dimension
        :arg nhead: number of Attention heads
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate"""
        super().__init__(*args, **kwargs)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.feedforward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """Compute the EncoderLayer forward.
        :arg src: source tensor of shape (N, S, E)
        :arg src_mask: mask tensor of shape (S, S)
        :return: tensor of shape (N, S, E)"""

        # Self-attention
        src2 = self.self_attn(src, src, src, mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feedforward
        src = src + self.feedforward(src)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    """Implement the Encoder class. The encoder is made up of a stack of N encoder layers."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, *args, **kwargs):
        """Initialize the Encoder class.
        :arg d_model: embedding
        :arg nhead: number of Attention heads
        :arg num_layers: number of encoder layers
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate"""
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """Compute the Encoder forward.
        :arg src: source tensor of shape (N, S, E)
        :arg src_mask: mask tensor of shape (S, S)
        :return: tensor of shape (N, S, E)"""
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src


class DecoderLayer(nn.Module):
    """Implement the DecoderLayer class. The decoder layer is made up of self-attn, cross-attn and feedforward
    network. """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, *args, **kwargs):
        """Initialize the DecoderLayer class.
        :arg d_model: embedding dimension
        :arg nhead: number of Attention heads
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate"""
        super().__init__(*args, **kwargs)
        # Attention layers
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.feedforward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masking
        mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1))).type(torch.bool)
        if tgt_mask is not None:
            tgt_mask = tgt_mask & mask.to(tgt_mask.device)

        if tgt_mask is None:
            tgt_mask = mask.to(tgt.device)

        # self attention
        self_attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(self_attn_output)
        tgt = self.norm1(tgt)

        # cross attention
        cross_attn_output = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_output)
        tgt = self.norm2(tgt)

        # feedforward
        tgt = tgt + self.feedforward(tgt)
        tgt = self.norm3(tgt)
        return tgt


class Decoder(nn.Module):
    """Implement the Decoder class. The decoder is made up of a stack of N decoder layers."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, *args, **kwargs):
        """Initialize the Decoder class.
        :arg d_model: embedding
        :arg nhead: number of Attention heads
        :arg num_layers: number of decoder layers
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate"""
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Compute the Decoder forward.
        :arg tgt: target tensor of shape (N, T, E)
        :arg memory: memory tensor of shape (N, S, E)
        :arg tgt_mask: mask tensor of shape (T, T)
        :arg memory_mask: mask tensor of shape (T, S)
        :return: tensor of shape (N, T, E)"""

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        tgt = self.norm(tgt)
        return tgt


class Transformer(nn.Module):
    """Implement the Transformer class. The transformer model is made up of a stack of N encoder and decoder layers."""

    def __init__(self, d_model, nhead, num_encoding_layers, num_decoding_layers, dim_feedforward, dropout=0.1, *args,
                 **kwargs):
        """Initialize the Transformer class.
        :arg d_model: embedding
        :arg nhead: number of Attention heads
        :arg num_encoding_layers: number of encoder layers
        :arg num_decoding_layers: number of decoder layers
        :arg dim_feedforward: dimension of the feedforward network
        :arg dropout: dropout rate
        """

        super().__init__(*args, **kwargs)
        self.pe = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(d_model, nhead, num_encoding_layers, dim_feedforward, dropout)
        self.decoder = Decoder(d_model, nhead, num_decoding_layers, dim_feedforward, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """Compute the Transformer forward.
        :arg src: source tensor of shape (N, S, E)
        :arg tgt: target tensor of shape (N, T, E)
        :arg src_mask: mask tensor of shape (S, S)
        :arg tgt_mask: mask tensor of shape (T, T)
        :arg memory_mask: mask tensor of shape (T, S)
        :return: tensor of shape (N, T, E)"""
        src = self.pe(src)
        tgt = self.pe(tgt)
        memory = self.encoder(src, src_mask)

        return self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask
        )

