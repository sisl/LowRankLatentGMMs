#*******************************************************************************
# imports and setup
#*******************************************************************************
import argparse
import logging
import matplotlib.pyplot as plt
import os
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from torchdyn.core import NeuralODE

# torchcfm imports
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchcfm.models.models import MLP

# file imports
from evaluation.mmd import mmd
from evaluation.cnf_metrics import cnf_test_metrics
from models.mppca import LowRankMixtureModel
from models.cnf import cnf_test_metrics, torch_wrapper
from utils.datasets import create_data_loaders
from utils.early_stopping import EarlyStopping
from utils.utils import plot_data, load_config


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str, default="normal",
                    choices=["normal", "mppca"])
parser.add_argument("--flow", type=str, default="cfm",
                    choices=["cfm", "otcfm"])
parser.add_argument("--dataset", type=str, default="power",
                    choices=["power", "gas", "hepmass", "miniboone", "bsds300"])
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# create results directory
model_dir = "./results/{}/{}".format(args.dataset, args.flow + "-" + args.base)
os.makedirs(model_dir, exist_ok=True)

# set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(model_dir, "training.log"), mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
if args.base == "mppca":
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
    if type(base) == LowRankMixtureModel:
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


#*******************************************************************************
# set up models and optimizers
#*******************************************************************************
# set up flow matcher model
if args.flow == "cfm":
    flow_matcher = ConditionalFlowMatcher(sigma=0.1)
elif args.flow == "otcfm":
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
else:
    raise ValueError

# define the Neural ODE network
model = MLP(dim=n_features, w=hidden_units, time_varying=True).to(device)
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Number of model parameters: {}".format(model_params))

# define training objects
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
total_steps = args.epochs * len(train_loader)
scheduler = CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
early_stopping = EarlyStopping(patience=args.patience, delta=1e-4, verbose=True)
logger.info(str(optimizer))


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
    mppca_lp = base_distribution.fit(
        x=data, 
        max_iterations=em_iters, 
        feature_sampling=False
    )
    end = time.time()
    logger.info("MPPCA fitting time: {:0.2f} s".format(end - start))
else:
    raise ValueError


#*******************************************************************************
# main training loop
#*******************************************************************************
torch.manual_seed(42)
logger.info("--------------------")
#lrs = []
for epoch in range(args.epochs):
    logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
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
            logger.info(
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

torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))

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

# read in all test data for maximum mean discrepancy calculation
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

# compute maximum mean discrepancy
maximum_mean_discrepancy = mmd(model_samples.cpu(), test_data)

std_ll, mean_ll = torch.std_mean(torch.tensor(avg_lps))
std_nfe, mean_nfe = torch.std_mean(torch.tensor(nfes))

logger.info(f"test log-likelihood: {mean_ll:.4f} ± {std_ll:.4f}")
logger.info(f"test NFE: {mean_nfe:.4f} ± {std_nfe:.4f}")
logger.info(f"test MMD: {maximum_mean_discrepancy:.8f}")

# plot samples vs test data
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
plot_data(5, test_data[:500], axes[:, :], color="b")
plot_data(5, model_samples[:500], axes[:, :], color="r")
plt.savefig(os.path.join(model_dir, "data_vs_samples.png"))
