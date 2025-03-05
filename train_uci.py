#*******************************************************************************
# imports and setup
#*******************************************************************************
# packages
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdyn.core import NeuralODE
from tqdm import tqdm

# torchcfm imports
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchcfm.models.models import MLP

# file imports
from evaluation.cnf_metrics import cnf_test_metrics
from models.mppca import MPPCA
from models.cnf import cnf_test_metrics, torch_wrapper
from utils.datasets import create_data_loaders
from utils.early_stopping import EarlyStopping
from utils.utils import plot_data, load_config


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="Normal",
                    choices=["Normal", "MPPCA"])
parser.add_argument("--flow", type=str, default="CFM",
                    choices=["CFM", "OTCFM"])
parser.add_argument("--dataset", type=str, default="power",
                    choices=["POWER", "GAS", "HEPMASS", "MINIBOONE", "BSDS300"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n_trials", type=int, default=2)
args = parser.parse_args()

# create results directory
results_dir = f"./results/{args.dataset}/{args.flow}-{args.base}"
os.makedirs(results_dir, exist_ok=True)

# set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# read in experiment hyperparameters
hyperparameters = load_config("experiments.json", args.dataset)

n_features = hyperparameters["n_features"]
n_components = hyperparameters["n_components"]
n_factors = hyperparameters["n_factors"]
em_iters = hyperparameters["em_iters"]
batch_size = hyperparameters["batch_size"]
learning_rate = hyperparameters["learning_rate"]
train_split = hyperparameters["train_split"]
val_split = hyperparameters["val_split"]
test_split = hyperparameters["test_split"]
hidden_units = hyperparameters["hidden_units"]


#*******************************************************************************
# read in data
#*******************************************************************************
# read in data
data_dir = "data/uci/"
data_path = os.path.join(data_dir, args.dataset + ".npy")

print("Reading in data...")
train_loader, val_loader, test_loader = create_data_loaders(data_path, batch_size)

# read in dataset as tensor dataset for fitting MPPCA base
if args.base == "MPPCA":
    data = []
    for batch in train_loader:
        data.append(batch)
    data = torch.cat(data, dim=0)


#*******************************************************************************
# utility functions
#*******************************************************************************
def sample_base(base, N):
    """
    Wrapper function to sample from both MPPCA models and torch distributions.

    Parameters:
    base (distribution): either LowRankMixtureModel() or torch distribution object
    N (int): total number of samples to draw

    Returns:
    samples (tensor): [N x D] tensor of generated samples
    """
    if type(base) == MPPCA:
        samples = base.sample(N, with_noise=True)[0]
    else:
        samples = base.sample((N,))
        
    return samples


def compute_loss(x0, x1, flow_matcher, model):
    """
    Compute the conditional flow matching loss.

    Parameters:
    x0 (tensor): [B x D] tensor of samples from the base distribution
    x1 (tensor): [B x D] tensor of samples from target distribution
    flow_matcher (ConditionalFlowMatcher): conditional flow matching object
    model (MLP): neural ODE model
    N (int): total number of samples to draw

    Returns:
    loss (tensor): scalar loss value
    """
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    return loss


def save_metrics(dir, log_probs, NFEs, total_epochs, base_fit_times, flow_train_times):
    metrics = {}
    metrics['log probs'] = np.array(log_probs).tolist()
    metrics['log probs mean'] = float(log_probs.mean())
    metrics['log probs std'] = float(log_probs.std())
    metrics['NFEs'] = np.array(NFEs).tolist()
    metrics['NFEs mean'] = float(NFEs.mean())
    metrics['NFEs std'] = float(NFEs.std())
    metrics['epochs'] = np.array(total_epochs).tolist()
    metrics['epochs mean'] = float(total_epochs.mean())
    metrics['epochs std'] = float(total_epochs.std())
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


log_probs = torch.zeros(args.n_trials)
NFEs = torch.zeros(args.n_trials)
total_epochs = torch.zeros(args.n_trials)
base_fit_times = torch.zeros(args.n_trials)
flow_train_times = torch.zeros(args.n_trials)


for trial in range(args.n_trials):
    #*******************************************************************************
    # set up models and optimizers
    #*******************************************************************************
    # set up flow matcher model
    if args.flow == "CFM":
        flow_matcher = ConditionalFlowMatcher(sigma=0.1)
    elif args.flow == "OTCFM":
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    else:
        raise ValueError

    # define the Neural ODE network
    model = MLP(dim=n_features, w=hidden_units, time_varying=True).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model parameters: {}".format(model_params))

    # define training objects
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
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
        mppca_lp = base_distribution.fit(
            x=data, 
            max_iterations=em_iters
        )
        end = time.time()
        base_fit_time = end - start
        print("MPPCA fitting time: {:0.2f} s".format(base_fit_time))
    else:
        raise ValueError


    #*******************************************************************************
    # main training loop
    #*******************************************************************************
    torch.manual_seed(trial)
    print("--------------------")
    start = time.time()
    epochs = 0
    for epoch in range(args.epochs):
        epochs += 1
        print(f"Starting epoch {epoch + 1}/{args.epochs}")
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x0 = sample_base(base=base_distribution, N=batch_size).to(device)
            x1 = batch.to(device)
            loss = compute_loss(x0, x1, flow_matcher, model)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()

            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Batch [{i + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing validation loss"):
                x0 = sample_base(base=base_distribution, N=batch_size).to(device)
                x1 = batch.to(device)
                val_loss += compute_loss(x0, x1, flow_matcher, model).item()
                
        val_loss /= len(val_loader)
        print("--------------------")
        print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss:.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Stopping early at epoch {epoch + 1}")
            break
        print("--------------------")

    end = time.time()
    flow_train_time = end - start
    print("Total training time: {:0.2f} s".format(flow_train_time))
    print("--------------------")

    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pt'))

    #*******************************************************************************
    # model evaluation
    #*******************************************************************************
    model.eval()

    node = NeuralODE(
        vector_field=torch_wrapper(model),  
        solver="dopri5", 
        sensitivity="adjoint", 
        atol=1e-4, 
        rtol=1e-4
    )

    avg_lps = []
    nfes = []
    for batch in tqdm(test_loader, desc="Computing test metrics"):
        test_samples = batch.to(device)
        avg_lp, nfe = cnf_test_metrics(model, test_samples, device, base_distribution)
        avg_lps.append(avg_lp)
        nfes.append(nfe)

    std_ll, mean_ll = torch.std_mean(torch.tensor(avg_lps))
    std_nfe, mean_nfe = torch.std_mean(torch.tensor(nfes))

    print(f"test log-likelihood: {mean_ll:.4f} ± {std_ll:.4f}")
    print(f"test NFE: {mean_nfe:.4f} ± {std_nfe:.4f}")

    # read in test data for plotting real vs. generated samples
    test_data = []
    for batch in test_loader:
        test_data.append(batch)
    test_data = torch.cat(test_data, dim=0)

    # generate samples from the learned model
    base_samples = sample_base(base=base_distribution, N=int(1e5)).to(device)
    with torch.no_grad():
        model_samples = node.trajectory(
            base_samples,
            t_span=torch.linspace(0, 1, 2, device=device),
        )[-1].cpu()

    # plot samples vs test data
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    plot_data(5, test_data[:500], axes[:, :], color="b")
    plot_data(5, model_samples[:500], axes[:, :], color="r")
    plt.savefig(os.path.join(results_dir, "data_vs_samples.png"))

    log_probs[trial] = mean_ll
    NFEs[trial] = mean_nfe
    total_epochs[trial] = epochs
    base_fit_times[trial] = base_fit_time
    flow_train_times[trial] = flow_train_time

    save_metrics(results_dir, log_probs, NFEs, total_epochs, base_fit_times, flow_train_times)