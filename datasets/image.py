import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
from torchvision.datasets import ImageFolder

from torchvision.datasets import CelebA, FGVCAircraft, FashionMNIST, CIFAR10
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
                transforms.Resize((32, 32)),
                #transforms.Resize(self.image_shape[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ReshapeTransform([-1])
            ])
        elif self.dataset == "cifar10":
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])
            mppca_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
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
        elif self.dataset == "cifar10":
            dataset = CIFAR10(
                root=self.root_dir,
                train=True,
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
    
    def get_dataloaders(self, batch_size):
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
        elif self.dataset == "cifar10":
            temp_dataset = CIFAR10(root=self.root_dir, train=True,
                transform=cfm_transforms, download=True
            )
            train_size = int(0.8 * len(temp_dataset))  # 80% for training
            val_size = len(temp_dataset) - train_size  # 20% for validation

            train_dataset, val_dataset = random_split(temp_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

            test_dataset = CIFAR10(root=self.root_dir, train=False,
                transform=cfm_transforms, download=True
            )
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
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        
        return train_loader, val_loader, test_loader