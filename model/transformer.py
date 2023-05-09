from torch import nn

from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


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

