# Inspired by the unofficial PyTorch implementation of Denoising Diffusion
# Probabilistic Models (https://github.com/w86763777/pytorch-ddpm/tree/master)
# and the TorchCFM repository 
# (https://github.com/atong01/conditional-flow-matching).

#*******************************************************************************
# imports and setup
#*******************************************************************************
import argparse
import copy
import json
import numpy as np
import os
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from tqdm import tqdm

# torchcfm imports
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# file imports
from models.mppca import MPPCA
from utils.utils import load_config, set_seed
from utils.ndb import NDB

from datasets.image import ImageDataset


def create_training_options():
    """ Parse arguments, load training configurations, and save data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Normal", choices=["Normal", "MPPCA"])
    parser.add_argument("--flow", type=str, default="CFM", choices=["OTCFM", "VPCFM"])
    parser.add_argument("--dataset", type=str, default="fashion", choices=["fashion", "cifar10", "celeba"])
    parser.add_argument("--n_factors", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--n_trials", type=int, default=2)
    opt = parser.parse_args()

    # read in experiment hyperparameters
    config = load_config("experiments.json", opt.dataset)

    if opt.n_factors is None:
        opt.n_factors = config["n_factors"]

    opt.image_shape = config["image_shape"]
    opt.n_components = config["n_components"]
    opt.em_iters = config["em_iters"]
    opt.batch_size = config["batch_size"]
    opt.em_batch_size = config["em_batch_size"]
    opt.learning_rate = config["learning_rate"]
    opt.num_channels = config["num_channels"]
    opt.num_res_blocks = config["num_res_blocks"]
    opt.channel_mult = config["channel_mult"]

    # create run directory
    opt.run_dir = f"./runs/{opt.dataset}/{opt.flow}-{opt.base}"
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


def save_metrics(dir, NDBs, NFEs, base_fit_times, flow_train_times):
    metrics = {}
    metrics['NDB/C'] = np.array(NDBs).tolist()
    metrics['NFEs'] = np.array(NFEs).tolist()
    metrics['base fit times'] = np.array(base_fit_times).tolist()
    metrics['flow train times'] = np.array(flow_train_times).tolist()

    total_train_times = np.add(np.array(base_fit_times), np.array(flow_train_times))
    metrics['total train times'] = total_train_times.tolist()

    # save results
    with open(os.path.join(dir, "results.json"), "w") as outfile:
        json.dump(metrics, outfile)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def main(opt):
    """ Main training loop.
    
    Args:
    opt (argparse.Namespace): The training options object.
    """
    # set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(os.path.join(opt.run_dir, "checkpoints"), exist_ok=True)


    warmup = 5000
    def warmup_lr(step):
        return min(step, warmup) / warmup

    # compute the number of features
    n_features = np.prod(opt.image_shape)

    #*******************************************************************************
    # read in data
    #*******************************************************************************
    data_handler = ImageDataset(dataset=opt.dataset, root_dir="./data", image_shape=opt.image_shape)
    mppca_dataset = data_handler.get_mppca_dataset()
    train_loader, val_loader, test_loader = data_handler.get_dataloaders(opt.batch_size)
    transform_mean, transform_std = data_handler.transform_mean, data_handler.transform_std

    NDBs = torch.zeros(opt.n_trials)
    NFEs = torch.zeros(opt.n_trials)
    base_fit_times = torch.zeros(opt.n_trials)
    flow_train_times = torch.zeros(opt.n_trials)

    print("\n****************************************")
    print(f"Training {opt.flow} model with {opt.base} base distribution.")
    print(f"Dataset: {opt.dataset} with {n_features} dimensions.")
    print("****************************************\n")

    for trial in range(opt.n_trials):
        set_seed(trial)
        #*******************************************************************************
        # set up models and optimizers
        #*******************************************************************************
        # set up flow matcher model
        if opt.flow == "OTCFM":
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        elif opt.flow == "VPCFM":
            flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.0)
        else:
            raise ValueError

        # define the Neural ODE network
        model = UNetModelWrapper(
            dim=(opt.image_shape[-1], opt.image_shape[0], opt.image_shape[1]),
            num_res_blocks=opt.num_res_blocks,
            num_channels=opt.num_channels,
            channel_mult=opt.channel_mult,
            num_heads=4,
            num_head_channels=opt.num_channels,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)

    #*******************************************************************************
    # construct base distribution
    #*******************************************************************************
    if opt.base == "Normal":
        base_distribution = MultivariateNormal(
            torch.zeros(n_features).to(device), 
            torch.eye(n_features).to(device)
        )
        base_fit_time = 0.0
    elif opt.base == "MPPCA":
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
    else:
        raise ValueError

    #*******************************************************************************
    # main training loop
    #*******************************************************************************
    print("----------------------------------------")
    start = time.time()
    epochs = 0
    for epoch in range(opt.epochs):
        epochs += 1
        print(f"Starting epoch {epoch + 1}/{opt.epochs}")
        ema_model.train()
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            x0 = sample_base(base=base_distribution, N=opt.batch_size, image_shape=opt.image_shape, with_noise=True).to(device)
            x1 = batch[0].to(device)
            loss = compute_loss(x0, x1, flow_matcher, model)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            if i % 16 == 0:
                ema(model, ema_model, 0.984)

        ema_model.eval()
        model.eval()

        # generate sample images to check training progress
        node_ = NeuralODE(ema_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            samples = sample_base(base=base_distribution, N=64, image_shape=opt.image_shape, with_noise=True).to(device)
            traj = node_.trajectory(
                samples.to(device),
                t_span=torch.linspace(0, 1, 2, device=device),
            )
            img = traj[-1, :].view([-1, opt.image_shape[-1], opt.image_shape[0], opt.image_shape[1]]).clip(-1,1)
            img = img * transform_std[:, None, None].to(device) + transform_mean[:, None, None].to(device)
            save_image(img, os.path.join(opt.run_dir, f"epoch_{epoch+1}.png"), nrow=8)

    end = time.time()
    flow_train_time = end - start
    print("Total training time: {:0.2f} s".format(flow_train_time))
    print("----------------------------------------")

    torch.save(ema_model.state_dict(), os.path.join(opt.run_dir, 'model.pt'))

    #*******************************************************************************
    # model evaluation
    #*******************************************************************************
    ema_model.eval()
    node = NeuralODE(ema_model,  solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    # read in all test data for NDB calculation
    n_bins = 200
    test_data = []
    for batch in test_loader:
        test_data.append(batch[0])
    test_data = torch.cat(test_data, dim=0)

    N = test_data.shape[0]
    test_data = test_data.reshape((N, -1))

    # generate samples from the learned model
    base_samples = sample_base(base=base_distribution, N=N, image_shape=opt.image_shape, with_noise=True).to(device)

    base_dataset = DataLoader(base_samples, batch_size=opt.batch_size)
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
    ndb_evaluator = NDB(training_data=test_data, number_of_bins=n_bins, significance_level=0.05, whitening=False)
    results = ndb_evaluator.evaluate(model_samples, 'Validation')

    print("NDB/K: {:0.4f}".format(results["NDB"]/n_bins))
    print("JS: {:0.8f}".format(results["JS"]))

    NDBs[trial] = results["NDB"]/n_bins

    # test NFE
    test_NFEs = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # node needs to be reset every time
            node = NeuralODE(ema_model,  solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            traj = node.trajectory(
                batch[0].to(device),
                t_span=torch.linspace(1, 0, 2, device=device),
            )
            NFE = node.vf.nfe
            test_NFEs.append(NFE)

    std_NFE, mean_NFE = torch.std_mean(torch.tensor(test_NFEs))
    print(f"test NFE: {mean_NFE:.4f} ± {std_NFE:.4f}")

    NFEs[trial] = mean_NFE

    base_fit_times[trial] = base_fit_time
    flow_train_times[trial] = flow_train_time

    save_metrics(opt.run_dir, NDBs, NFEs, base_fit_times, flow_train_times)
    time.sleep(2)


if __name__ == "__main__":
    opt = create_training_options()
    main(opt)