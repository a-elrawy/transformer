from torch import nn

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

