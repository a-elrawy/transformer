import urllib
import os
from io import open
import torch

from torch.utils.data import Dataset, DataLoader


class ShakespearDataset(Dataset):
    def __init__(self, file_path='data/shakespeare.txt', max_seq_len=64, tokenizer=None):
        if not os.path.exists(file_path):
            # Download the dataset if it does not exist
            os.makedirs("data", exist_ok=True)
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, file_path)
            print(f'Downloaded {url} to {file_path}')

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


def train_test_split(dataset, train_size=0.8):
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def get_shakespeare_dataloader(batch_size, max_seq_len=64, tokenizer=None):
    dataset = ShakespearDataset(file_path='data/shakespeare.txt', max_seq_len=max_seq_len, tokenizer=tokenizer)

    train_dataset, test_dataset = train_test_split(dataset)
    train_dataset, val_dataset = train_test_split(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
