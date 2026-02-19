import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CelebA, FashionMNIST, CIFAR10
import torchvision.transforms as transforms

from utils.utils import CropTransform, ReshapeTransform


class ImageDataset:
    def __init__(self, dataset, root_dir, image_shape):
        """
        Initializes the CelebAHandler class.

        Args:
            root_dir (str): Path to the directory containing the CelebA dataset.
            image_size (int): Desired size of the output image (image will be resized).
        """    
        self.dataset = dataset
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.mppca_transforms = self.get_transforms()


    def get_transforms(self):
        # image transformations before fitting MPPCA model
        if "celeba" in self.dataset:
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            mppca_transforms = transforms.Compose([
                CropTransform((25, 50, 25+128, 50+128)), 
                transforms.Resize(self.image_shape[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ReshapeTransform([-1])
            ])
        elif "cifar10" in self.dataset:
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            mppca_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ReshapeTransform([-1])
            ])
        elif "fashion" in self.dataset:
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
        # we check this way because there maybe a series
        # of different "celeba-*"s which scales based on
        # how much we want to upscale the images
        if "celeba" in self.dataset:
            dataset = CelebA(
                root=self.root_dir,
                split="train",
                transform=self.mppca_transforms,
                download=True
            )
        elif "cifar10" in self.dataset:
            dataset = CIFAR10(
                root=self.root_dir,
                train=True,
                transform=self.mppca_transforms,
                download=True
            )
        elif "fashion" in self.dataset:
            dataset = FashionMNIST(
                root=self.root_dir,
                train=True,
                transform=self.mppca_transforms,
                download=True
            )
        else:
            raise ValueError(f"Dataset '{self.dataset}' is invalid.")
        
        return dataset
    
    def get_dataloaders(self, batch_size):
        cfm_transforms = transforms.Compose(self.mppca_transforms.transforms[:-1])

        if "celeba" in self.dataset:
            train_dataset = CelebA(root=self.root_dir, split="train",
                transform=cfm_transforms, download=True)
            val_dataset = CelebA(root=self.root_dir, split="valid",
                transform=cfm_transforms, download=True)  
            test_dataset = CelebA(root=self.root_dir, split="test",
                transform=cfm_transforms, download=True)  
        elif "cifar10" in self.dataset:
            temp_dataset = CIFAR10(root=self.root_dir, train=True,
                transform=cfm_transforms, download=True
            )
            train_size = int(0.8 * len(temp_dataset))  # 80% for training
            val_size = len(temp_dataset) - train_size  # 20% for validation

            train_dataset, val_dataset = random_split(temp_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

            test_dataset = CIFAR10(root=self.root_dir, train=False,
                transform=cfm_transforms, download=True
            )
        elif "fashion" in self.dataset:
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
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        
        return train_loader, val_loader, test_loader
