#%%
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.models import MLP

from cnf import autograd_trace, hutch_trace, compute_log_probs
from models import LowRankMixtureModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='POWER',
                    choices=['POWER', 'GAS', 'HEPMASS', 'MINIBOONE', 'BSDS300'])
args = parser.parse_args()
dataset_name = args.dataset

#dataset_name = "BSDS300"
if dataset_name == "POWER":
    n_features = 6
    n_components = 10
    n_factors = 3
    batch_size = 128
    num_training_steps = 100000
elif dataset_name == "GAS":
    n_features = 8
    n_components = 10
    n_factors = 3
    batch_size = 128
    num_training_steps = 100000
elif dataset_name == "HEPMASS":
    n_features = 21
    n_components = 10
    n_factors = 4
    batch_size = 128
    num_training_steps = 100000
elif dataset_name == "MINIBOONE":
    n_features = 43
    n_components = 30
    n_factors = 6
    batch_size = 64
    num_training_steps = 100000
elif dataset_name == "BSDS300":
    n_features = 63
    n_components = 30
    n_factors = 6
    batch_size = 128
    num_training_steps = 100000
else:
    assert False, 'Unknown dataset: ' + dataset_name


class UCIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def my_infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x

def sample_base(base, N, with_noise):
    if type(base) == LowRankMixtureModel:
        samples = base.sample(N, with_noise=with_noise)[0]
    else:
        samples = base.sample((N,))
        
    return samples

def plot_data(n_features, X, axes, color=None):
    """
    Plot samples from an MPPCA model.

    Parameters:
    n_features (int): number of input dimensions (alias: d)
    X (torch.Tensor): [n x d] tensor of data samples
    axes (np.array): array of matplotlib Axes objects
    color (str): hex color code
    """
    for i in range(n_features):
        for j in range(n_features):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_box_aspect(1)
            if i == j:
                axes[i, j].text(0.5, 0.5, f'Dim {i+1}', ha='center', va='center', fontsize=12)
            else:
                axes[i, j].scatter(X[:, j], X[:, i], alpha=0.5, color=color)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)


model_dir = './models/' + dataset_name
os.makedirs(model_dir, exist_ok=True)

data_dir = "./data/UCI/"
data = torch.tensor(np.load(data_dir + dataset_name + ".npy"), dtype=torch.float32)

# Create the dataset
dataset = UCIDataset(data)

# Optional: Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
datalooper = my_infiniteloop(dataloader)


init_method = "kmeans"
max_iterations = 50
feature_sampling = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device="cpu"

#%%
model = LowRankMixtureModel(
    n_components=n_components,
    n_features=n_features,
    n_factors=n_factors,
    init_method=init_method
).to(device)

ll_log = model.fit(data, max_iterations=max_iterations, feature_sampling=feature_sampling)
#ll_log = model.batch_fit(dataset, max_iterations=max_iterations, batch_size=256, feature_sampling=feature_sampling)

print('Saving the model...')
torch.save(model.state_dict(), os.path.join(model_dir, 'mppca_'+ dataset_name + '.pth'))

# plot log-likelihood
plt.plot(ll_log, c = "b")
plt.xlabel("EM iteration")
plt.ylabel("Log-Likelihood")
plt.savefig("uci/EM_log_likelihood_{}.png".format(dataset_name))

# plot sample comparison
n_plot = 1000
samples, _ = model.sample(n_plot, with_noise=True)
samples = samples.to('cpu')
plot_features = 5
fig, axes = plt.subplots(plot_features, plot_features, figsize=(15, 15))
plot_data(plot_features, data[:n_plot], axes[:, :], color="b")
plot_data(plot_features, samples[:n_plot], axes[:, :],  color="r")
plt.savefig("uci/uci_eval_{}.png".format(dataset_name))


#%%
def train_fm_model(model, FM, base, optimizer, scheduler, batch_size, note):
    start = time.time()
    losses = []
    log_probs = []
    for k in range(num_training_steps):
        optimizer.zero_grad()

        x0 = sample_base(base=base, N=batch_size, with_noise=True)
        x0 = x0.to(device)
        x1 = next(datalooper).to(device)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()

        if (k + 1) % 200 == 0:

            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
        
            avg_lp = compute_log_probs(model, x1, 1, device, base, autograd_trace).mean().cpu()

            losses.append(loss.item())
            log_probs.append(avg_lp)

    torch.save(model, os.path.join(model_dir, 'flow_'+ note + '.pth'))

    return model, losses, log_probs

#%%
sigma = 0.1
base = MultivariateNormal(
    torch.zeros(n_features).to(device), 
    torch.eye(n_features).to(device)
)
model = MLP(dim=n_features, w=128, time_varying=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, num_training_steps, 0)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.requires_grad)
print(f"The optimizer is optimizing {total_params} parameters.")

note = "normal"
model_normal, losses_normal, log_probs_normal = train_fm_model(
    model, FM, base, optimizer, scheduler, batch_size, note)


# %%
print('Loading pre-trained MPPCA model...')
model_dir = './models/' + dataset_name
model_dict = torch.load(os.path.join(model_dir, 'mppca_'+ dataset_name + '.pth'), weights_only=True)
n_components, n_features, n_factors = model_dict['W'].shape
base_mppca = LowRankMixtureModel(
    n_components=n_components,
    n_features=n_features,
    n_factors=n_factors
)
base_mppca.load_state_dict(model_dict)
base_mppca.to(device)

sigma = 0.1
model_mppca = MLP(dim=n_features, w=128, time_varying=True).to(device)
optimizer_mppca = torch.optim.SGD(model_mppca.parameters(), lr=5e-4, weight_decay=1e-6)

total_params = sum(p.numel() for group in optimizer_mppca.param_groups for p in group['params'] if p.requires_grad)

print(f"The optimizer is optimizing {total_params} parameters.")

scheduler_mppca = CosineAnnealingLR(optimizer_mppca, num_training_steps, 0)
FM_mppca = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

note = "mppca"
model_mppca, losses_mppca, log_probs_mppca = train_fm_model(
    model_mppca, FM_mppca, base_mppca, optimizer_mppca, scheduler_mppca, batch_size, note)

# %%
plt.figure()
plt.plot(losses_mppca, label="MPPCA")
plt.plot(losses_normal,label="Normal")
plt.legend()
plt.savefig("uci/losses_{}.png".format(dataset_name))

plt.figure()
plt.plot(log_probs_mppca, label="MPPCA")
plt.plot(log_probs_normal, label="Normal")
plt.legend()
plt.savefig("uci/log_probs_{}.png".format(dataset_name))


# %%
