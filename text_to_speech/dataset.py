import os

import librosa
from torch.utils.data import Dataset

from utils import compute_spectrogram, get_dataloaders, pad_collate
from utils.file_processing import download_and_extract_dataset


class LJDataset(Dataset):
    def __init__(self, root_dir, max_text_len=1000, tokenizer=None):
        self.root_dir = root_dir
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        # Download the dataset if it does not exist
        download_and_extract_dataset('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2', root_dir)
        self.metadata = []
        with open(os.path.join(root_dir, 'metadata.csv'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                wav_file = os.path.join(root_dir, 'wavs', f'{parts[0]}.wav')
                text = parts[2][:max_text_len].lower()
                self.metadata.append((wav_file, tokenizer.encode(text)))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_file, text = self.metadata[idx]

        # Load audio file
        audio, sr = librosa.load(wav_file, sr=22050)
        # Compute spectrogram
        logspec = compute_spectrogram(audio, sr)
        return text, logspec


def get_LJ_dataloader(batch_size, max_seq_len=1000, tokenizer=None):
    dataset = LJDataset(root_dir='data/LJSpeech-1.1', max_text_len=max_seq_len, tokenizer=tokenizer)
    return get_dataloaders(dataset, batch_size=batch_size, collate_fn=pad_collate)
