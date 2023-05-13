from torch import nn

from model.feed_forward import FeedForward
from model.attention import MultiHeadAttention


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
        src2 = self.self_attn(src, src, src, mask=src_mask)
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
