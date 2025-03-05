# Inspired by the unofficial PyTorch implementation of Denoising Diffusion
# Probabilistic Models (https://github.com/w86763777/pytorch-ddpm/tree/master)
# and the TorchCFM repository 
# (https://github.com/atong01/conditional-flow-matching).

#%%
#*******************************************************************************
# imports and setup
#*******************************************************************************
import argparse
import copy
import logging
import numpy as np
import os
from PIL import Image
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdyn.core import NeuralODE
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm


# torchcfm imports
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper

# file imports
from models.mppca import LowRankMixtureModel
from utils.datasets import ImageDataset
from utils.early_stopping import EarlyStopping
from utils.utils import load_config


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="Normal",
                    choices=["Normal", "MPPCA"])
parser.add_argument("--flow", type=str, default="CFM",
                    choices=["CFM", "OTCFM"])
parser.add_argument("--dataset", type=str, default="power",
                    choices=["POWER", "GAS", "HEPMASS", "MINIBOONE", "BSDS300"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n_trials", type=int, default=2)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create results directory
results_dir = "./results/{}/{}".format(args.dataset, args.flow + "-" + args.base)
os.makedirs(results_dir, exist_ok=True)


# read in experiment hyperparameters
hyperparameters = load_config("experiments.json", args.dataset)

image_shape = hyperparameters["image_shape"]
n_components = hyperparameters["n_components"]
n_factors = hyperparameters["n_factors"]
em_iters = hyperparameters["em_iters"]
batch_size = hyperparameters["batch_size"]
em_batch_size = hyperparameters["em_batch_size"]
learning_rate = hyperparameters["learning_rate"]
num_channels = hyperparameters["num_channels"]

# compute the number of features
n_features = np.prod(image_shape)

#*******************************************************************************
# read in data
#*******************************************************************************
data_handler = ImageDataset(dataset=args.dataset, root_dir="./data", batch_size=batch_size, image_shape=image_shape)

mppca_dataset = data_handler.get_mppca_dataset()

train_loader, val_loader, test_loader = data_handler.get_dataloaders()


transform_mean, transform_std = data_handler.transform_mean, data_handler.transform_std


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
    if type(base) == LowRankMixtureModel:
        samples = base.sample(N, with_noise=with_noise)[0].view(
            N, image_shape[-1], image_shape[0], image_shape[1])
    else:
        samples = base.sample((N,)).view(
            N, image_shape[-1], image_shape[0], image_shape[1])
        
    return samples


def compute_loss(x0, x1, flow_matcher, model):
    """
    Compute the conditional flow matching loss.

    Parameters:
    x0 (tensor): [B x D] tensor of samples from the base distribution
    x1 (tensor): [B x D] tensor of samples from target distribution
    flow_matcher (ConditionalFlowMatcher): conditional flow matching object
    model (U-Net): neural ODE model

    Returns:
    loss (tensor): scalar loss value
    """
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
    vt = model(t, xt)
    loss = torch.mean((vt - ut) ** 2)

    return loss

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

#*******************************************************************************
# set up models and optimizers
#*******************************************************************************
# set up flow matcher model
if args.flow == "cfm":
    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
elif args.flow == "otcfm":
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
else:
    raise ValueError

# define the Neural ODE network
'''
model = UNetModelWrapper(
    dim=(3, 64, 64),
    num_res_blocks=2,
    num_channels=64,
    channel_mult=[1, 2, 3, 4],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)
'''
model = UNetModelWrapper(
    dim=(1, 28, 28),
    num_res_blocks=1,
    num_channels=32,
    channel_mult=[1, 2, 2],
    num_heads=4,
    num_head_channels=32,
    attention_resolutions="16",
    dropout=0.1,
).to(device)

ema_model = copy.deepcopy(model)

# show NODE model size
model_size = 0
for param in model.parameters():
    model_size += param.data.nelement()
print("Number of model parameters: %.2f M" % (model_size / 1024 / 1024))

# define training objects
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-6)
total_steps = args.epochs * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
early_stopping = EarlyStopping(patience=args.patience, delta=1e-4, verbose=True)


#*******************************************************************************
# construct base distribution
#*******************************************************************************
start = time.time()

if args.base == "normal":
    base_distribution = MultivariateNormal(
        torch.zeros(n_features).to(device), 
        torch.eye(n_features).to(device)
    )
elif args.base == "mppca":
    base_distribution = LowRankMixtureModel(
        n_components=n_components,
        n_features=n_features,
        n_factors=n_factors,
        init_method="kmeans"
    ).to(device)
    # count MPPCA parameters
    mppca_params = int(n_components*(n_features*n_factors+n_features+1)+(n_components-1))
    logger.info("Number of MPPCA parameters: {}".format(mppca_params))
    mppca_lp = base_distribution.batch_fit(
        train_dataset=mppca_dataset, 
        batch_size=em_batch_size, 
        max_iterations=em_iters,
        feature_sampling=False)
    end = time.time()
    logger.info(f"Number of MPPCA batch EM iterations: {em_iters}")
    logger.info(f"Final log-likelihood: {mppca_lp[-1]:.4f}")
    logger.info("MPPCA fitting time: {:0.2f} s".format(end - start))
else:
    raise ValueError


#*******************************************************************************
# main training loop
#*******************************************************************************
torch.manual_seed(42)
logger.info("--------------------")
for epoch in range(args.epochs):
    logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
    ema_model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        #with torch.autocast("cuda"):
        optimizer.zero_grad()
        x0 = sample_base(base=base_distribution, N=batch_size, image_shape=image_shape, with_noise=True).to(device)
        x1 = batch[0].to(device)
        loss = compute_loss(x0, x1, flow_matcher, model)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        ema(model, ema_model, 0.9999)
        '''
        if (i + 1) % 20 == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{args.epochs}], "
                f"Batch [{i + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        '''
    ema_model.eval()

    # generate sample images to check training progress
    node_ = NeuralODE(ema_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        samples = sample_base(base=base_distribution, N=64, image_shape=image_shape, with_noise=True).to(device)
        traj = node_.trajectory(
            samples.to(device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )
        img = traj[-1, :]#.view([-1, 3, 64, 64])
        img = img * transform_std[:, None, None].to(device) + transform_mean[:, None, None].to(device)
        save_image(img, os.path.join(model_dir, f"epoch_{epoch+1}.png"), nrow=8)

    # compute validation loss
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing validation loss"):
            x0 = sample_base(base=base_distribution, N=batch_size, image_shape=image_shape, with_noise=True).to(device)
            x1 = batch[0].to(device)
            val_loss += compute_loss(x0, x1, flow_matcher, ema_model).item()
            
    val_loss /= len(val_loader)
    logger.info("--------------------")
    logger.info(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss:.4f}")
    early_stopping(val_loss, logger)
    if early_stopping.early_stop:
        logger.info(f"Stopping early at epoch {epoch + 1}")
        break
    logger.info("--------------------")

end = time.time()

logger.info("Total training time: {:0.2f} s".format(end - start))
logger.info("--------------------")

torch.save(ema_model.state_dict(), os.path.join(results_dir, 'model.pt'))

#%%
#*******************************************************************************
# model evaluation
#*******************************************************************************
ema_model.eval()

node = NeuralODE(
    vector_field=ema_model,  
    solver="dopri5", 
    sensitivity="adjoint", 
    atol=1e-4, 
    rtol=1e-4
)

from ndb import *
from torch.utils.data import DataLoader

# read in all test data for maximum mean discrepancy calculation
test_data = []
for batch in test_loader:
    test_data.append(batch[0])
test_data = torch.cat(test_data, dim=0)

N = test_data.shape[0]

test_data = test_data.reshape((N, -1))

# generate samples from the learned model
base_samples = sample_base(base=base_distribution, N=N, image_shape=image_shape, with_noise=True).to(device)

base_dataset = DataLoader(base_samples, batch_size=batch_size)


model_samples = []
for batch in tqdm(base_dataset):
    with torch.no_grad():
        transformed_data = node.trajectory(
            batch,
            t_span=torch.linspace(0, 1, 2, device=device),
        )[-1].cpu()
    model_samples.append(transformed_data)

model_samples = torch.cat(model_samples, dim=0)

model_samples = model_samples.reshape((N, -1))

# compute NDB
mnist_ndb = NDB(training_data=test_data, number_of_bins=200, significance_level=0.05, whitening=False)


results = mnist_ndb.evaluate(model_samples, 'Validation')

logger.info("NDB/K: {:0.4f}".format(results["NDB"]/200))
logger.info("JS: {:0.8f}".format(results["JS"]))


nfes = []
for batch in tqdm(test_loader):
    with torch.no_grad():
        node = NeuralODE(
            vector_field=ema_model,  
            solver="dopri5", 
            sensitivity="adjoint", 
            atol=1e-4, 
            rtol=1e-4
        )
        traj = node.trajectory(
            batch[0].to(device),
            t_span=torch.linspace(1, 0, 2, device=device),
        )
        nfe = node.vf.nfe
        nfes.append(nfe)

std_nfe, mean_nfe = torch.std_mean(torch.tensor(nfes))
logger.info(f"test NFE: {mean_nfe:.4f} ± {std_nfe:.4f}")
time.sleep(2)