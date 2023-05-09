import numpy as np
import torch
import unittest

from transformers import GPT2Tokenizer

from text_to_speech import TTSModel, TextToSpeech


class TestTextToSpeech(unittest.TestCase):
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
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = TTSModel(self.tokenizer.vocab_size, self.d_model, self.nhead, self.num_encoding_layers,
                              self.num_decoding_layers,
                              self.dim_feedforward, self.dropout, self.input_dim, self.output_dim)
        self.tts = TextToSpeech(self.model, self.tokenizer)

    def test_train(self):
        train_data_text = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(2, 32))
        train_data_audio = torch.randint(low=-100, high=0, size=(2, 1000, self.input_dim), dtype=torch.float32)
        val_data_text = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(2, 32))
        val_data_audio = torch.randint(low=-100, high=0, size=(2, 1000, self.input_dim), dtype=torch.float32)
        train_dataloader = torch.utils.data.DataLoader(list(zip(train_data_text, train_data_audio)), batch_size=2)
        val_dataloader = torch.utils.data.DataLoader(list(zip(val_data_text, val_data_audio)), batch_size=2)

        self.tts.train(train_dataloader, val_dataloader, epochs=2)

    def test_evaluate(self):
        test_data_text = torch.randint(low=0, high=self.tokenizer.vocab_size, size=(16, 32))
        test_data_audio = torch.randint(low=-100, high=0, size=(16, 1000, self.input_dim), dtype=torch.float32)
        test_dataloader = torch.utils.data.DataLoader(list(zip(test_data_text, test_data_audio)), batch_size=16)

        loss = self.tts.evaluate(test_dataloader)
        self.assertIsInstance(loss, float)

    def test_generate(self):
        context = "The quick brown fox jumps over the lazy dog"
        generated = self.tts.generate(context, max_len=100)
        self.assertIsInstance(generated, np.ndarray)
