from io import open

import torch
from torch.utils.data import Dataset

from utils.data_loader import get_dataloaders
from utils.file_processing import download_file


class ShakespearDataset(Dataset):
    def __init__(self, file_path='data/shakespeare.txt', max_seq_len=64, tokenizer=None):
        # Download the dataset if it does not exist
        download_file('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                      file_path)

        self.file_path = file_path
        self.data = self._load_data()
        self.tokenizer = tokenizer

        # Encode the data into a sequence of tokens
        self.tokens = self.tokenizer.encode(self.data)
        # Split the data into chunks of length max_seq_len
        self.tokens = [self.tokens[i:i + max_seq_len] for i in range(0, len(self.tokens), max_seq_len)]
        # Remove the last chunk if it is not of length max_seq_len
        if len(self.tokens[-1]) != max_seq_len:
            self.tokens = self.tokens[:-1]

    def __len__(self):
        return len(self.tokens)

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            data = f.read()
        return data

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx])


def get_shakespeare_dataloader(batch_size, max_seq_len=64, tokenizer=None):
    dataset = ShakespearDataset(file_path='data/shakespeare.txt', max_seq_len=max_seq_len, tokenizer=tokenizer)
    return get_dataloaders(dataset, batch_size=batch_size)
