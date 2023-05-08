import torch
import unittest

from text_prediction import GPT, TextPredictor
from transformers import GPT2Tokenizer


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
