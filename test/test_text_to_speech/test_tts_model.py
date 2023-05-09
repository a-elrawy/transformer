import torch
import unittest

from text_to_speech import TTSModel


class TestTTSModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 500
        self.d_model = 512
        self.nhead = 4
        self.num_encoding_layers = 6
        self.num_decoding_layers = 6
        self.dim_feedforward = 1024
        self.dropout = 0.1
        self.input_dim = 80
        self.output_dim = 80
        self.model = TTSModel(self.vocab_size, self.d_model, self.nhead, self.num_encoding_layers, self.num_decoding_layers,
                         self.dim_feedforward, self.dropout)

    def test_forward(self):
        text = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        spec = torch.randint(low=-100, high=0, size=(16, 1000, self.input_dim), dtype=torch.float32)

        output = self.model(text, spec)
        self.assertEqual(output.shape, (16, 1000, self.output_dim))

