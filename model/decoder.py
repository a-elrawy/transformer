from torch import nn
import torch

from model.feed_forward import FeedForward
from model.attention import MultiHeadAttention

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
