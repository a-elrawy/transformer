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
