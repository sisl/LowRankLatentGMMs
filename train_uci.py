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
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher
)
from torchcfm.models.models import MLP

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# file imports
from models.mppca import MPPCA
from models.cnf import cnf_test_metrics, torch_wrapper
from utils.ckpt_utils import load_best_checkpoint, save_checkpoint
from utils.early_stopping import EarlyStopping
from utils.utils import plot_data, load_config, set_seed
from datasets.uci import UCIDataset


def create_training_options():
    """ Parse arguments, load training configurations, and save data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Normal", choices=["Normal", "MPPCA"])
    parser.add_argument("--flow", type=str, default="CFM", choices=["OTCFM", "VPCFM"])
    parser.add_argument("--dataset", type=str, default="power", choices=["HEPMASS", "MINIBOONE", "BSDS300"])
    parser.add_argument("--n_factors", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=3)
    opt = parser.parse_args()

    # read in experiment hyperparameters
    config = load_config("experiments.json", opt.dataset)

    if opt.n_factors is None:
        opt.n_factors = config["n_factors"]

    opt.n_features = config["n_features"]
    opt.n_components = config["n_components"]
    opt.em_iters = config["em_iters"]
    opt.batch_size = config["batch_size"]
    opt.learning_rate = config["learning_rate"]
    opt.hidden_units = config["hidden_units"]

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
    metrics['NFEs'] = np.array(NFEs).tolist()
    metrics['epochs'] = np.array(total_epochs).tolist()
    metrics['base fit times'] = np.array(base_fit_times).tolist()
    metrics['flow train times'] = np.array(flow_train_times).tolist()

    total_train_times = np.add(np.array(base_fit_times), np.array(flow_train_times))
    metrics['total train times'] = total_train_times.tolist()

    # save results
    with open(os.path.join(dir, "results.json"), "w") as outfile:
        json.dump(metrics, outfile)


def main(opt):
    """ Main training loop.
    
    Args:
    opt (argparse.Namespace): The training options object.
    """
    # set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(os.path.join(opt.run_dir, "checkpoints"), exist_ok=True)
    
    #*******************************************************************************
    # read in data
    #*******************************************************************************
    # read in data
    data_dir = "data/uci/"
    data_path = os.path.join(data_dir, opt.dataset + ".npy")

    print("Reading in data...")
    dataset = UCIDataset(data_path)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(opt.batch_size)

    # read in dataset as tensor dataset for fitting MPPCA base
    if opt.base == "MPPCA":
        data = []
        for batch in train_loader:
            data.append(batch)
        data = torch.cat(data, dim=0)

    # Tensors to hold results
    log_probs = torch.zeros(opt.n_trials)
    NFEs = torch.zeros(opt.n_trials)
    total_epochs = torch.zeros(opt.n_trials)
    base_fit_times = torch.zeros(opt.n_trials)
    flow_train_times = torch.zeros(opt.n_trials)


    print("\n****************************************")
    print(f"Training {opt.flow} model with {opt.base} base distribution.")
    print(f"Dataset: {opt.dataset} with {opt.n_features} dimensions.")
    print("****************************************\n")
    for trial in range(opt.n_trials):
        set_seed(trial)
        #***************************************************************************
        # set up models and optimizers
        #***************************************************************************
        # set up flow matcher model
        if opt.flow == "OTCFM":
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
        elif opt.flow == "VPCFM":
            flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
        else:
            raise ValueError

        # define the Neural ODE network
        model = MLP(dim=opt.n_features, w=opt.hidden_units, time_varying=True).to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of model parameters: {}".format(model_params))

        # define training objects
        #optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=1e-6)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        total_steps = opt.epochs * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
        early_stopper = EarlyStopping(patience=opt.patience, delta=1e-4)
                
        #***************************************************************************
        # construct base distribution
        #***************************************************************************
        if opt.base == "Normal":
            base_distribution = MultivariateNormal(
                torch.zeros(opt.n_features).to(device), 
                torch.eye(opt.n_features).to(device)
            )
            base_fit_time = 0.0
        elif opt.base == "MPPCA":
            start = time.time()
            base_distribution = MPPCA(
                n_components=opt.n_components,
                n_features=opt.n_features,
                n_factors=opt.n_factors
            ).to(device)
            # count MPPCA parameters
            mppca_params = int(opt.n_components*(opt.n_features*opt.n_factors+opt.n_features+1)+(opt.n_components-1))
            print("Number of MPPCA parameters: {}".format(mppca_params))
            mppca_lp = base_distribution.fit(
                x=data, 
                max_iterations=opt.em_iters
            )
            end = time.time()
            base_fit_time = end - start
            print("MPPCA fitting time: {:0.2f} s".format(base_fit_time))
        else:
            raise ValueError

        #***************************************************************************
        # main training loop
        #***************************************************************************
        print("----------------------------------------")
        start = time.time()
        epochs = 0
        for epoch in range(opt.epochs):
            epochs += 1
            print(f"Starting epoch {epoch + 1}/{opt.epochs}")
            model.train()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x0 = sample_base(base=base_distribution, N=opt.batch_size).to(device)
                x1 = batch.to(device)
                loss = compute_loss(x0, x1, flow_matcher, model)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()

                if (i + 1) % 200 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{opt.epochs}], "
                        f"Batch [{i + 1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}"
                    )

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Computing validation loss"):
                    x0 = sample_base(base=base_distribution, N=opt.batch_size).to(device)
                    x1 = batch.to(device)
                    val_loss += compute_loss(x0, x1, flow_matcher, model).item()
                    
            val_loss /= len(val_loader)
            print("----------------------------------------")
            print(f"Epoch {epoch + 1}/{opt.epochs}, Val Loss: {val_loss:.4f}")

            # Save checkpoints and check for early stopping
            save_checkpoint(opt.run_dir, epoch+1, model, optimizer, scheduler, val_loss, best=False)

            if early_stopper.step(val_loss):
                save_checkpoint(opt.run_dir, epoch+1, model, optimizer, scheduler, val_loss, best=True)

            if early_stopper.should_stop:
                print(f"Stopping early at epoch {epoch + 1}")
                break

            print("----------------------------------------")

        end = time.time()
        flow_train_time = end - start
        print("Total training time: {:0.2f} s".format(flow_train_time))
        print("----------------------------------------")

        #***************************************************************************
        # model evaluation
        #***************************************************************************
        load_best_checkpoint(opt.run_dir, model)

        model.eval()

        node = NeuralODE(
            vector_field=torch_wrapper(model),  
            solver="dopri5", 
            sensitivity="adjoint", 
            atol=1e-4, 
            rtol=1e-4
        )

        avg_log_probs = []
        avg_NFEs = []
        for batch in tqdm(test_loader, desc="Computing test metrics"):
            test_samples = batch.to(device)
            avg_log_prob, avg_NFE = cnf_test_metrics(model, test_samples, device, base_distribution)
            avg_log_probs.append(avg_log_prob)
            avg_NFEs.append(avg_NFE)

        std_log_prob, mean_log_prob = torch.std_mean(torch.tensor(avg_log_probs))
        std_NFE, mean_NFE = torch.std_mean(torch.tensor(avg_NFEs))

        print(f"test log-likelihood: {mean_log_prob:.4f} ± {std_log_prob:.4f}")
        print(f"test NFE: {mean_NFE:.4f} ± {std_NFE:.4f}")

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
        plt.savefig(os.path.join(opt.run_dir, "data_vs_samples.png"))

        log_probs[trial] = mean_log_prob
        NFEs[trial] = mean_NFE
        total_epochs[trial] = epochs
        base_fit_times[trial] = base_fit_time
        flow_train_times[trial] = flow_train_time

        save_metrics(opt.run_dir, log_probs, NFEs, total_epochs, base_fit_times, flow_train_times)


if __name__ == "__main__":
    opt = create_training_options()
    main(opt)