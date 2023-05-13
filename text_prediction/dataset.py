from io import open

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
        self.sentences = self.data.split('.')
        # Remove empty lines
        self.sentences = [s for s in self.sentences if len(s) > 0]
        self.tokenizer = tokenizer
        # Add PAD token to tokenizer if it does not exist
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sentences)

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            data = f.read()
        return data

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx], return_tensors='pt', max_length=self.max_seq_len,
                                padding='max_length', truncation=True)
        # Remove batch dimension
        tokens = {key: val.squeeze(0) for key, val in tokens.items()}
        return tokens


def get_shakespeare_dataloader(batch_size, max_seq_len=64, tokenizer=None):
    dataset = ShakespearDataset(file_path='data/shakespeare.txt', max_seq_len=max_seq_len, tokenizer=tokenizer)
    return get_dataloaders(dataset, batch_size=batch_size)
