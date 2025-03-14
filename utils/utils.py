import matplotlib.pyplot as plt
import numpy as np
import json
import sklearn.datasets as datasets
import torch

from models.mppca import MPPCA

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def generate_data(target, rng=None, batch_size=256):
    if rng is None:
        rng = np.random.RandomState()

    if target == "moons":
        X = datasets.make_moons(n_samples=batch_size, noise=0.08)[0]
        X = X.astype("float32")
        X = X * 2 + np.array([-1, -0.2])
        return X

    elif target == "swissroll":
        X = datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        X = X.astype("float32")[:, [0, 2]]
        X /= 5
        return X

    elif target == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0), 
            (-1, 0), 
            (0, 1), 
            (0, -1), 
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)), 
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        X = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            X.append(point)
        X = np.array(X, dtype="float32")
        X /= 1.414
        return X

    elif target == "checkerboard":
        X1 = np.random.rand(batch_size) * 4 - 2
        X2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        X2 = X2_ + (np.floor(X1) % 2)
        return np.concatenate([X1[:, None], X2[:, None]], 1) * 2

    elif target == "pinwheel":
        radial_std = 0.3
        
        
def plot_data(n_features, X, axes, color=None):
    """
    Plot samples from an MPPCA model.

    Parameters:
    n_features (int): number of input dimensions (alias: d)
    X (torch.Tensor): [n x d] tensor of data samples
    axes (np.array): array of matplotlib Axes objects
    color (str): hex color code
    """
    for i in range(n_features):
        for j in range(n_features):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_box_aspect(1)
            if i == j:
                axes[i, j].text(0.5, 0.5, f'Dim {i+1}', ha='center', va='center', fontsize=12)
            else:
                axes[i, j].scatter(X[:, j], X[:, i], alpha=0.5, color=color)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


def load_config(config_file, dataset_name):
    """
    Loads dataset-specific configuration from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.
        dataset_name (str): Name of the dataset.

    Returns:
        dict: Hyperparameters for the specified dataset.
    """
    with open(config_file, "r") as file:
        config = json.load(file)

    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration file.")

    return config[dataset_name]