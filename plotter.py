#%%
# this is to verify sampling with / without noise, I think
# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import argparse
import copy
import os
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import trange

from PIL import Image

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

# file imports
from models import LowRankMixtureModel
from utils import CropTransform, infiniteloop, sample_base, samples_to_mosaic

args = argparse.Namespace()
args.base = "mppca"
args.model_file = "model_c_250_l_10.pth"
args.num_channel = 64
args.lr = 2e-4
args.grad_clip = 1.0
args.total_steps = 200000
args.warmup = 1000
args.batch_size = 64
args.save_step = 2000

# image shape [H, W, n_channels]
image_shape = [32, 32, 3]
n_features = np.prod(image_shape)

model_dir = './models/celeba/'
figure_dir = './figures/celeba/flow_{}/'.format(args.base)
os.makedirs(figure_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define base distribution
if args.base == "mppca":
    print('Loading pre-trained MPPCA model...')
    model_dict = torch.load(model_dir + args.model_file, weights_only=True)
    n_components, n_features, n_factors = model_dict['W'].shape
    base = LowRankMixtureModel(
        n_components=n_components,
        n_features=n_features,
        n_factors=n_factors
    )
    base.load_state_dict(model_dict)
    base.to(device)
else:
    base = MultivariateNormal(
        torch.zeros(n_features).to(device), 
        torch.eye(n_features).to(device)
    )

#%%
print('Generating random samples...')
rnd_samples, _ = base.sample(100, with_noise=False)
mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
image = Image.fromarray((255 * mosaic).astype(np.uint8))
image.save('samples1.png')



# samples = sample_base(base=base, N=64, image_shape=image_shape, with_noise=False)
# save_image(traj, figure_dir + f"{args.base}_base_step_{step}.png", nrow=8)
# %%
