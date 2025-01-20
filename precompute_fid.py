#%%
from torchvision import transforms
import os
from utils.utils import CropTransform

import torch
import numpy as np

import cleanfid

#import cleanfid
from cleanfid.fid import get_folder_features
#from cleanfid.utils import *
from cleanfid.features import build_feature_extractor
#from cleanfid.resize import *


def precompute_fid(name, fdir, num_workers=0, batch_size=64, transform=None):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    mode = "clean"

    outf = os.path.join(stats_folder, f"{name}_{mode}_{split}_{res}.npz".lower())

    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)

    feat_model = build_feature_extractor(mode)

    # get all inception features for folder images
    np_feats = get_folder_features(fdir, feat_model, num_workers=num_workers,
                                    batch_size=batch_size,
                                    mode=mode, description=f"custom stats ({name}):",
                                    custom_fn_resize=transform)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")

    np.savez_compressed(outf, mu=mu, sigma=sigma)


#*******************************************************************************
# Pre-compute FID for CelebA
#*******************************************************************************
image_shape = [32, 32, 3]
transform = transforms.Compose(
    [
        CropTransform((25, 50, 25+128, 50+128)), 
        transforms.Resize(image_shape[0]),
        transforms.ToTensor()
    ]
)

from PIL import Image
def transform_and_load(image_path):
    image = Image.fromarray(image_path)
    #img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = np.array(transform(image))
    return img


# Path to your dataset
dataset_path = "./data/celeba/img_align_celeba/"
# Name for the custom statistics
custom_stats_name = "celeba"

# Generate and save custom statistics
precompute_fid(
    name=custom_stats_name,  # Name of the custom stats file
    fdir=dataset_path,  # Path to your dataset
    num_workers=4,  # Number of workers for parallel processing
    transform=transform_and_load  # Apply your custom transformation
)

print(f"Custom statistics saved for {custom_stats_name}.")


#*******************************************************************************
# Pre-compute FID for FGVC Aircraft
#*******************************************************************************
image_shape = [32, 32, 3]
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(image_shape[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)

from PIL import Image
def transform_and_load(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = Image.fromarray(transform(img))
    return img


# Path to your dataset
dataset_path = "./data/fgvc-aircraft-2013b/data/images/"
# Name for the custom statistics
custom_stats_name = "fgvc-aircraft"

# Generate and save custom statistics
precompute_fid(
    name=custom_stats_name,  # Name of the custom stats file
    fdir=dataset_path,  # Path to your dataset
    num_workers=4,  # Number of workers for parallel processing
    transform=transform_and_load  # Apply your custom transformation
)

print(f"Custom statistics saved for {custom_stats_name}.")
