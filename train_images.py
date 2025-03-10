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
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper

# file imports
from models.mppca import MPPCA
from utils.early_stopping import EarlyStopping
from utils.utils import load_config
from utils.ndb import NDB

from datasets.image import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="Normal",
                    choices=["Normal", "MPPCA"])
parser.add_argument("--flow", type=str, default="CFM",
                    choices=["CFM", "OTCFM", "VPCFM"])
parser.add_argument("--dataset", type=str, default="fashion",
                    choices=["fashion", "celeba", "fgvc-aircraft", "cifar10"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--n_trials", type=int, default=1)
args = parser.parse_args()

# KEEP 20
# 
warmup = 5000
def warmup_lr(step):
    return min(step, warmup) / warmup

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
num_res_blocks = hyperparameters["num_res_blocks"]
channel_mult = hyperparameters["channel_mult"]

# compute the number of features
n_features = np.prod(image_shape)

#*******************************************************************************
# read in data
#*******************************************************************************
data_handler = ImageDataset(dataset=args.dataset, root_dir="./data", image_shape=image_shape)

mppca_dataset = data_handler.get_mppca_dataset()

train_loader, val_loader, test_loader = data_handler.get_dataloaders(batch_size)


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
    metrics['NDB/C mean'] = float(NDBs.mean())
    metrics['NDB/C std'] = float(NDBs.std())
    metrics['NFEs'] = np.array(NFEs).tolist()
    metrics['NFEs mean'] = float(NFEs.mean())
    metrics['NFEs std'] = float(NFEs.std())
    metrics['base fit times'] = np.array(base_fit_times).tolist()
    metrics['base fit times mean'] = float(base_fit_times.mean())
    metrics['base fit times std'] = float(base_fit_times.std())
    metrics['flow train times'] = np.array(flow_train_times).tolist()
    metrics['flow train times mean'] = float(flow_train_times.mean())
    metrics['flow train times std'] = float(flow_train_times.std())

    total_train_times = np.add(np.array(base_fit_times), np.array(flow_train_times))
    metrics['total train times'] = total_train_times.tolist()
    metrics['total train times mean'] = float(total_train_times.mean())
    metrics['total train times std'] = float(total_train_times.std())

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

NDBs = torch.zeros(args.n_trials)
NFEs = torch.zeros(args.n_trials)
base_fit_times = torch.zeros(args.n_trials)
flow_train_times = torch.zeros(args.n_trials)

print("\n****************************************")
print(f"Training {args.flow} model with {args.base} base distribution.")
print(f"Dataset: {args.dataset} with {n_features} dimensions.")
print("****************************************\n")

for trial in range(args.n_trials):
    torch.manual_seed(trial)
    #*******************************************************************************
    # set up models and optimizers
    #*******************************************************************************
    # set up flow matcher model
    if args.flow == "CFM":
        flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    elif args.flow == "OTCFM":
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    elif args.flow == "VPCFM":
        flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.0)
    else:
        raise ValueError

    # define the Neural ODE network

    model = UNetModelWrapper(
        dim=(image_shape[-1], image_shape[0], image_shape[1]),
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        channel_mult=channel_mult,
        num_heads=4,
        num_head_channels=num_channels,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_steps = args.epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)
    early_stopping = EarlyStopping(patience=args.patience, delta=1e-4, verbose=True)


    #*******************************************************************************
    # construct base distribution
    #*******************************************************************************
    if args.base == "Normal":
        base_distribution = MultivariateNormal(
            torch.zeros(n_features).to(device), 
            torch.eye(n_features).to(device)
        )
        base_fit_time = 0.0
    elif args.base == "MPPCA":
        start = time.time()
        base_distribution = MPPCA(
            n_components=n_components,
            n_features=n_features,
            n_factors=n_factors
        ).to(device)
        # count MPPCA parameters
        mppca_params = int(n_components*(n_features*n_factors+n_features+1)+(n_components-1))
        print("Number of MPPCA parameters: {}".format(mppca_params))
        mppca_lp = base_distribution.batch_fit(
            train_dataset=mppca_dataset, 
            batch_size=em_batch_size, 
            max_iterations=em_iters)
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
    for epoch in range(args.epochs):
        epochs += 1
        print(f"Starting epoch {epoch + 1}/{args.epochs}")
        ema_model.train()
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            x0 = sample_base(base=base_distribution, N=batch_size, image_shape=image_shape, with_noise=True).to(device)
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
            samples = sample_base(base=base_distribution, N=64, image_shape=image_shape, with_noise=True).to(device)
            traj = node_.trajectory(
                samples.to(device),
                t_span=torch.linspace(0, 1, 2, device=device),
            )
            img = traj[-1, :].view([-1, image_shape[-1], image_shape[0], image_shape[1]]).clip(-1,1)
            img = img * transform_std[:, None, None].to(device) + transform_mean[:, None, None].to(device)
            save_image(img, os.path.join(results_dir, f"epoch_{epoch+1}.png"), nrow=8)

        '''
        # compute validation loss
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing validation loss"):
                x0 = sample_base(base=base_distribution, N=batch_size, image_shape=image_shape, with_noise=True).to(device)
                x1 = batch[0].to(device)
                val_loss += compute_loss(x0, x1, flow_matcher, model).item()
                
        val_loss /= len(val_loader)
        print("--------------------")
        print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Stopping early at epoch {epoch + 1}")
            break
        print("--------------------")
        '''

    end = time.time()
    flow_train_time = end - start
    print("Total training time: {:0.2f} s".format(flow_train_time))
    print("----------------------------------------")

    torch.save(ema_model.state_dict(), os.path.join(results_dir, 'model.pt'))

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

    save_metrics(results_dir, NDBs, NFEs, base_fit_times, flow_train_times)
    time.sleep(2)