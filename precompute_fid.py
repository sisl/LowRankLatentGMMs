#%%
from cleanfid import fid
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from utils.utils import CropTransform, ReshapeTransform

import os
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from PIL import Image
from scipy import linalg
import zipfile
import cleanfid

import cleanfid
from cleanfid.fid import get_folder_features
from cleanfid.utils import *
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.resize import *

# Pre-compute FID for CelebA
image_shape = [32, 32, 3]
transform = transforms.Compose(
    [
        CropTransform((25, 50, 25+128, 50+128)), 
        transforms.Resize(image_shape[0]),
        #transforms.ToTensor(),  
        ReshapeTransform([-1])
    ]
)
# Define a custom function for loading and transforming images
def transform_and_load(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = Image.fromarray(transform(img))
    return img

# Path to your dataset
dataset_path = "./data/celeba/img_align_celeba/"

# Name for the custom statistics
custom_stats_name = "celeba"



def my_make_custom_stats(name, fdir, num=None, mode="clean", model_name="inception_v3",
                    num_workers=0, batch_size=64, device=torch.device("cuda"), verbose=True,
                    custom_fn_resize=None):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    if model_name=="inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_"+model_name
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower())
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)
    if model_name=="inception_v3":
        feat_model = build_feature_extractor(mode, device)
        custom_fn_resize = None
        custom_image_tranform = None
    elif model_name=="clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32")
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
        custom_image_tranform = None
    else:
        raise ValueError(f"The entered model name - {model_name} was not recognized.")

    # get all inception features for folder images
    np_feats = get_folder_features(fdir, feat_model, num_workers=num_workers, num=num,
                                    batch_size=batch_size, device=device, verbose=verbose,
                                    mode=mode, description=f"custom stats: {os.path.basename(fdir)} : ",
                                    custom_image_tranform=custom_image_tranform,
                                    custom_fn_resize=custom_fn_resize)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom FID stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)
    
    # KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz".lower())
    print(f"saving custom KID stats to {outf}")
    np.savez_compressed(outf, feats=np_feats)


#%%
# Generate and save custom statistics
my_make_custom_stats(
    name=custom_stats_name,  # Name of the custom stats file
    fdir=dataset_path,  # Path to your dataset
    num_workers=4,  # Number of workers for parallel processing
    mode = 'clean',
    custom_fn_resize=transform_and_load  # Apply your custom transformation
)

print(f"Custom statistics saved as: {custom_stats_name}")

# %%
from cleanfid.fid import remove_custom_stats
remove_custom_stats(custom_stats_name, mode="clean", model_name="inception_v3")