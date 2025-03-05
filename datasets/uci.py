import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class UCIDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the .npy file containing the dataset.
        """
        self.data = np.load(file_path).astype(np.float32)  # Ensure data is in float32 format
        self.data = torch.tensor(self.data)  # Convert to PyTorch tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dataloaders(self, batch_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
        """
        Creates PyTorch DataLoaders for train, validation, and test splits.

        Args:
            file_path (str): Path to the .npy file containing the dataset.
            batch_size (int): Batch size for DataLoaders.
            train_split (float): Proportion of data to use for training.
            val_split (float): Proportion of data to use for validation.
            test_split (float): Proportion of data to use for testing.
            shuffle (bool): Whether to shuffle the dataset before splitting.

        Returns:
            tuple: DataLoaders for training, validation, and testing.
        """
        # Ensure splits sum to 1
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1."

        # Calculate split lengths
        total_len = len(self)
        train_len = int(train_split * total_len)
        val_len = int(val_split * total_len)
        test_len = total_len - train_len - val_len  # Remaining for test set

        # Split the dataset
        train_set, val_set, test_set = random_split(self, [train_len, val_len, test_len])

        # Create DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader