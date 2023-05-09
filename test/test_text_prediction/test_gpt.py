import torch
import unittest

from text_prediction import GPT


class TestGPT(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 512
        self.nhead = 8
        self.num_encoding_layers = 0
        self.num_decoding_layers = 12
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.model = GPT(self.vocab_size, self.d_model, self.nhead, self.num_encoding_layers, self.num_decoding_layers,
                         self.dim_feedforward, self.dropout)

    def test_forward(self):
        src = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        tgt = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        output = self.model(src, tgt)
        self.assertEqual(output.shape, (16, 32, self.vocab_size))

    def test_generate(self):
        src = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        generated = self.model.generate(src, max_len=100, max_seq_len=32)
        self.assertEqual(generated.shape, (16, 132))
        self.assertTrue((generated[:, :32] == src).all())

