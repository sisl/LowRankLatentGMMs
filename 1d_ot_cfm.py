import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import time
from tqdm import tqdm
import torch
from torch.distributions import Categorical, MixtureSameFamily, Normal

# torchdyn imports
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.models import MLP
from torchcfm.utils import torch_wrapper

# file imports
from cnf import CNF

# parse inputs
parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='normal',
                    choices=['normal', 'mppca'])
parser.add_argument('--cmap', type=str, default='inferno',
                    choices=['inferno', 'viridis', 'blue'])
parser.add_argument('--aspect', type=float, default=0.5)
args = parser.parse_args()

# create a colormap that matches presentation theme
if args.cmap == "blue":
    args.cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["black", "#0096DB"])

# define problem parameters
sigma = 0.001
dim = 1
batch_size = 128
n_iters = 2000
n_steps = 51

# base distribution
base = Normal(loc=0.0, scale=1.0)
# target distribution
means = torch.tensor([-1.0, 1.0])
stds = torch.tensor([0.5, 0.5])
mixing_probs = torch.ones(means.shape[0])/means.shape[0]
target = MixtureSameFamily(
    Categorical(probs=mixing_probs),
    Normal(loc=means, scale=stds)
)

# flow matching model
model = MLP(dim=dim, w=32, time_varying=True)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# main training loop
start = time.time()
for k in range(n_iters):
    optimizer.zero_grad()
    x0 = base.sample((batch_size,))[:,None]
    x1 = target.sample((batch_size,))[:,None]
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)
    loss.backward()
    optimizer.step()

    if (k + 1) % 50 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.4f} time {(end - start):0.4f}")
        start = end

# generate trajectories
n_samples = 21
tspan = torch.linspace(0, 1, n_steps)
nde = NeuralODE(DEFunc(torch_wrapper(model)), 
                solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)

base_samples = torch.linspace(-2, 2, n_samples)[:,None]
with torch.no_grad():
    trajectories = nde.trajectory(base_samples, t_span=tspan)

# helper function to compute log-likelihood values
def compute_trajectory_log_probs(model, trajectories):
    cnf = DEFunc(CNF(model))
    nde = NeuralODE(cnf, solver="euler", sensitivity="autograd")
    cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)

    lps = torch.zeros(n_steps, n_samples)
    for i, t in enumerate(tqdm(tspan)):
        with torch.no_grad():
            if t > 0:
                aug_traj = (cnf_model[1].trajectory(
                        Augmenter(1, 1)(trajectories[i,...]), 
                        t_span=torch.linspace(t, 0, n_steps)))[-1].cpu()
                log_probs = base.log_prob(aug_traj[:, 1]) - aug_traj[:, 0]
            else:
                log_probs = base.log_prob(base_samples.squeeze())

            lps[i] = log_probs.squeeze()

    return lps

# compute log-likelihoods
lps = compute_trajectory_log_probs(model, trajectories)

# plot trajectories
def plot_trajectories(trajectories, lps, ax):
    probs = lps.exp()
    vmax = base.log_prob(torch.tensor(0.0)).exp()
    norm = Normalize(vmin=probs.min(), vmax=vmax)

    for i in range(n_samples):
        x = trajectories[:,i].squeeze()
        points = np.array([x.numpy(), tspan.numpy()]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
        lc = LineCollection(segments, cmap=args.cmap, norm=norm)
        lc.set_array(probs[:,i])
        lc.set_linewidth(1)
        ax.add_collection(lc)

    ax.set_facecolor('black')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(args.aspect)

# plot PDF
def plot_pdf(distribution, ax):
    x = torch.linspace(-2, 2, 500)
    probs = distribution.log_prob(x).exp()
    norm = Normalize(vmin=probs.min(), vmax=probs.max())

    points = np.array([x.numpy(), probs.numpy()]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)

    lc = LineCollection(segments, cmap=args.cmap, norm=norm)
    lc.set_array(probs)
    lc.set_linewidth(1)
    ax.add_collection(lc)

    ax.set_facecolor('black')
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 0.6)
    ax.set_aspect(4.0)

fig = plt.figure()
# need box_aspect / height to be 6/10
height = args.aspect / (6./10.)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, height, 1], hspace=0.0)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

plot_pdf(target, ax1)
plot_trajectories(trajectories, lps, ax2)
plot_pdf(base, ax3)

plt.tight_layout()

plt.savefig("figures/{}.png".format(args.base), 
            dpi=600, bbox_inches='tight', pad_inches=0.0)