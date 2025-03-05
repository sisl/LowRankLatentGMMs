import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
from torchvision.datasets import ImageFolder

from torchvision.datasets import CelebA, FGVCAircraft, FashionMNIST
import torchvision.transforms as transforms

from .utils import CropTransform, ReshapeTransform

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


def create_data_loaders(file_path, batch_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
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

    # Load the dataset
    dataset = UCIDataset(file_path)

    # Calculate split lengths
    total_len = len(dataset)
    train_len = int(train_split * total_len)
    val_len = int(val_split * total_len)
    test_len = total_len - train_len - val_len  # Remaining for test set

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


class ImageDataset:
    def __init__(self, dataset, root_dir, batch_size, image_shape):
        """
        Initializes the CelebAHandler class.

        Args:
            root_dir (str): Path to the directory containing the CelebA dataset.
            batch_size (int): Number of samples per batch.
            image_size (int): Desired size of the output image (image will be resized).
        """    
        self.dataset = dataset
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.mppca_transforms = self.get_transforms()


    def get_transforms(self):
        # image transformations before fitting MPPCA model
        if self.dataset == "celeba":
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            mppca_transforms = transforms.Compose([
                CropTransform((25, 50, 25+128, 50+128)), 
                transforms.Resize(self.image_shape[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ReshapeTransform([-1])
            ])
        elif self.dataset == "fgvc-aircraft":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            mppca_transforms = transforms.Compose([
                transforms.Resize((64, 64)),
                #transforms.Resize(self.image_shape[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ReshapeTransform([-1])
            ])
        elif self.dataset == "fashion":
            mean = torch.tensor([0.0, 0.0, 0.0])
            std = torch.tensor([1.0, 1.0, 1.0])
            mppca_transforms = transforms.Compose([
                transforms.ToTensor(),
                ReshapeTransform([-1])
            ])
        else:
            raise ValueError(f"Dataset '{self.dataset}' is invalid.")

        self.transform_mean = mean
        self.transform_std = std

        return mppca_transforms


    def get_mppca_dataset(self):
        if self.dataset == "celeba":
            dataset = CelebA(
                root=self.root_dir,
                split="train",
                transform=self.mppca_transforms,
                download=True
            )
        elif self.dataset == "fgvc-aircraft":
            dataset = FGVCAircraft(
                root=self.root_dir,
                split="trainval",
                transform=self.mppca_transforms,
                download=True
            )
        elif self.dataset == "fashion":
            dataset = FashionMNIST(
                root=self.root_dir,
                train=True,
                transform=self.mppca_transforms,
                download=True
            )
        else:
            raise ValueError(f"Dataset '{self.dataset}' is invalid.")
        
        return dataset
    
    def get_dataloaders(self):
        cfm_transforms = transforms.Compose(self.mppca_transforms.transforms[:-1])

        if self.dataset == "celeba":
            train_dataset = CelebA(root=self.root_dir, split="train",
                transform=cfm_transforms, download=True)
            val_dataset = CelebA(root=self.root_dir, split="valid",
                transform=cfm_transforms, download=True)  
            test_dataset = CelebA(root=self.root_dir, split="test",
                transform=cfm_transforms, download=True)  
            
        elif self.dataset == "fgvc-aircraft":
            train_dataset = FGVCAircraft(root=self.root_dir, split="trainval",
                transform=cfm_transforms, download=True)
            
            temp_dataset = FGVCAircraft(root=self.root_dir, split="test",
                transform=cfm_transforms, download=True)

            val_size = int(0.5 * len(temp_dataset))  # 50% for validation
            test_size = len(temp_dataset) - val_size  # 50% for testing

            val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], torch.Generator().manual_seed(42))

        elif self.dataset == "fashion":
            temp_dataset = FashionMNIST(root=self.root_dir, train=True,
                transform=cfm_transforms, download=True
            )
            train_size = int(0.8 * len(temp_dataset))  # 80% for training
            val_size = len(temp_dataset) - train_size  # 20% for validation

            train_dataset, val_dataset = random_split(temp_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

            test_dataset = FashionMNIST(root=self.root_dir, train=False,
                transform=cfm_transforms, download=True
            )
        else:
            raise ValueError(f"Dataset '{self.dataset}' is invalid.")
        

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    
    '''
    def get_dataloaders(self, split='train'):
        cfm_transforms = transforms.Compose(self.mppca_transforms.transforms[:-1])

        if self.dataset == "celeba":
            dataset = CelebA(
                root=self.root_dir,
                split=split,
                transform=cfm_transforms,
                download=True
            )
        elif self.dataset == "fgvc-aircraft":
            dataset = FGVCAircraft(
                root=self.root_dir,
                split=split,
                transform=cfm_transforms,
                download=True
            )
        else:
            raise ValueError(f"Dataset '{self.dataset}' is invalid.")
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),  # Shuffle only for training split.
            drop_last=True
        )

        return dataloader
    '''

