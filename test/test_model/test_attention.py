import unittest
import torch

from model import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.d_model = 512
        self.nhead = 8
        self.batch_size = 4
        self.max_seq_len = 4

        self.multi_head_attention = MultiHeadAttention(
            self.d_model, self.nhead
        )

    def test_forward(self):
        q = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        v = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        output = self.multi_head_attention(q, k, v)
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

    def test_forward_with_mask(self):
        q = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        v = torch.randn(self.batch_size, self.max_seq_len, self.d_model)
        mask = torch.ones(self.batch_size, self.max_seq_len).bool()
        mask[3, 1:] = False
        output = self.multi_head_attention(q, k, v, mask=mask)
        expected_shape = (self.batch_size, self.max_seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        # Make sure the output is not all zeros
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
