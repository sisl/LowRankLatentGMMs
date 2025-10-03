# Inspired by the unofficial PyTorch implementation of Denoising Diffusion
# Probabilistic Models (https://github.com/w86763777/pytorch-ddpm/tree/master)
# and the TorchCFM repository 
# (https://github.com/atong01/conditional-flow-matching).

#*******************************************************************************
# imports and setup
#*******************************************************************************
import argparse
import json
import numpy as np
import os
import time
import torch
from torchvision.utils import save_image

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# file imports
from models.mppca import MPPCA
from utils.utils import load_config, set_seed

from datasets.image import ImageDataset


def create_training_options():
    """ Parse arguments, load training configurations, and save data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_factors", type=int, default=10)
    parser.add_argument("--n_components", type=int, default=50)
    opt = parser.parse_args()

    opt.image_shape = [128, 128, 3]
    opt.em_iters = 10
    opt.em_batch_size = 1000

    # create run directory
    opt.run_dir = f"./runs/mppca-scalability/mppca-l{opt.n_factors}-k{opt.n_components}"
    os.makedirs(opt.run_dir, exist_ok=True)

    # Save training options
    opt_file = os.path.join(opt.run_dir, "options.txt")
    with open(opt_file, 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    return opt


#*******************************************************************************
# utility functions
#*******************************************************************************
def sample_base(base, N, image_shape, with_noise=True):
    """
    Wrapper function to sample from both MPPCA models and torch distributions.

    Parameters:
    base (distribution): either LowRankMixtureModel() or torch distribution object
    N (int): total number of samples to draw

    Returns:
    samples (tensor): [N x D] tensor of generated samples
    """
    if type(base) == MPPCA:
        samples = base.sample(N, with_noise=with_noise)[0].view(
            N, image_shape[-1], image_shape[0], image_shape[1])
    else:
        samples = base.sample((N,)).view(
            N, image_shape[-1], image_shape[0], image_shape[1])
        
    return samples


def main(opt):
    """ Main training loop.
    
    Args:
    opt (argparse.Namespace): The training options object.
    """
    # set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # compute the number of features
    n_features = np.prod(opt.image_shape)

    #*******************************************************************************
    # read in data
    #*******************************************************************************
    data_handler = ImageDataset(dataset="celeba", root_dir="./data", image_shape=opt.image_shape)

    mppca_dataset = data_handler.get_mppca_dataset()
    transform_mean, transform_std = data_handler.transform_mean, data_handler.transform_std


    set_seed(42)

    start = time.time()
    base_distribution = MPPCA(
        n_components=opt.n_components,
        n_features=n_features,
        n_factors=opt.n_factors
    ).to(device)
    # count MPPCA parameters
    mppca_params = int(opt.n_components*(n_features*opt.n_factors+n_features+1)+(opt.n_components-1))
    print("Number of MPPCA parameters: {}".format(mppca_params))
    mppca_lp = base_distribution.batch_fit(
        train_dataset=mppca_dataset, 
        batch_size=opt.em_batch_size, 
        max_iterations=opt.em_iters)
    end = time.time()
    base_fit_time = end - start
    print(f"Final log-likelihood: {mppca_lp[-1]:.4f}")
    print("MPPCA fitting time: {:0.2f} s".format(end - start))


    #with torch.no_grad():
    samples = sample_base(base=base_distribution, N=64, image_shape=opt.image_shape, with_noise=True).to(device)

    img = samples.view([-1, opt.image_shape[-1], opt.image_shape[0], opt.image_shape[1]]).clip(-1,1)
    img = img * transform_std[:, None, None].to(device) + transform_mean[:, None, None].to(device)
    save_image(img, os.path.join(opt.run_dir, "samples.png"), nrow=8)



if __name__ == "__main__":
    opt = create_training_options()
    main(opt)