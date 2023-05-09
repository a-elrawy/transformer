import torch
import unittest

from text_prediction import GPT, TextPredictor
from transformers import GPT2Tokenizer


class TestTextPredictor(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.nhead = 8
        self.num_encoding_layers = 6
        self.num_decoding_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size

        self.model = GPT(self.vocab_size, self.d_model, self.nhead, self.num_encoding_layers, self.num_decoding_layers,
                         self.dim_feedforward, self.dropout)


        self.device = 'cpu'
        self.text_predictor = TextPredictor(self.model, self.tokenizer, self.device)

    def test_evaluate(self):
        input_data = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        dataloader = torch.utils.data.DataLoader(input_data, batch_size=16)
        loss = self.text_predictor.evaluate(dataloader)
        self.assertIsInstance(loss, float)

    def test_train(self):
        train_data = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        val_data = torch.randint(low=0, high=self.vocab_size, size=(16, 32))
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=16)

        self.text_predictor.train(train_dataloader, val_dataloader, epochs=2)

    def test_generate(self):
        context = "The quick brown fox jumps over the lazy dog"
        generated_text = self.text_predictor.generate(context, max_len=100, max_seq_len=32)
        self.assertIsInstance(generated_text, str)
