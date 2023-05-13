import torch
from torch.utils.data import DataLoader


def train_test_split(dataset, train_size=0.8):
    """Split a dataset into train and test sets.
    :param dataset: torch.utils.data.Dataset
    :param train_size: float between 0 and 1
    :return: train_dataset, test_dataset"""
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def get_dataloaders(dataset, batch_size=32, collate_fn=None):
    """Create train, validation and test dataloaders.
    :param dataset: torch.utils.data.Dataset
    :param batch_size: int
    :param collate_fn: function to collate data
    :return: train_dataloader, val_dataloader, test_dataloader"""
    train_dataset, test_dataset = train_test_split(dataset)
    train_dataset, val_dataset = train_test_split(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def pad_collate(batch):
    """Collate a batch data.
    :param batch: list of (src, target) tuples
    :return: padded srcs, padded targets"""
    (srcs, targets) = zip(*batch)

    srcs = [torch.tensor(src) for src in srcs]
    targets = [torch.tensor(target) for target in targets]

    # Pad to the same length
    srcs = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return srcs, targets
