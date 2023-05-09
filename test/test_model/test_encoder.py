import unittest
import torch

from model import Encoder


class TestEncoder(unittest.TestCase):

    def setUp(self):
        self.d_model = 512
        self.nhead = 8
        self.num_encoding_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.batch_size = 4
        self.max_seq_len = 4

        self.encoder = Encoder(
            self.d_model, self.nhead, self.num_encoding_layers,
            self.dim_feedforward, self.dropout
        )

    def test_forward(self):
        src = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        output = self.encoder(src=src)
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

    def test_forward_with_mask(self):
        src = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        src_mask = torch.ones(self.max_seq_len, self.max_seq_len).bool()
        src_mask[3, 1:] = False
        output = self.encoder(src, src_mask)
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        # Make sure the output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
