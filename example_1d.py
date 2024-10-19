#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize
import numpy as np
import os
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

from cnf import CNF


def rejection_sample(distribution, size, lower=-2, upper=2):
    samples = []
    while len(samples) < size:
        sample = distribution.sample((1,))  # Sample from standard normal
        if (sample >= lower) & (sample <= upper):  # Filter valid samples
            samples.append(sample)
    return torch.cat(samples)  # Concatenate all valid samples


#%%
sigma = 0.01
dim = 1
batch_size = 128
n_iters = 5000


# base
base = Normal(loc=0.0, scale=1.0)
# target
#means = torch.tensor([-1.0, 0.5, 1.5])
means = torch.tensor([-1.0, 1.0])
#stds = torch.tensor([0.5, 0.25, 0.25])
stds = torch.tensor([0.5, 0.5])
mixing_probs = torch.ones(means.shape[0])/means.shape[0]
target = MixtureSameFamily(
    Categorical(probs=mixing_probs),
    Normal(loc=means, scale=stds)
)


#%%
# flow matching model
model = MLP(dim=dim, w=32, time_varying=True)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

def train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note):

    savedir = "models/{}-1d".format(note)
    os.makedirs(savedir, exist_ok=True)

    start = time.time()

    for k in range(n_iters):
        optimizer.zero_grad()

        x0 = base.sample((batch_size,))[:,None]
        x1 = target.sample((batch_size,))[:,None]

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        #x1 = target.sample((batch_size,))
        #t, xtrJ = model(x1)
        #logprob = base.log_prob(xtrJ[1, :,1:]) - xtrJ[1, :,0]
        #loss = -torch.mean(logprob)
    
        loss.backward()
        optimizer.step()

        if (k + 1) % 20 == 0:

            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end


#%%
note = "normal-FM"

# if regular flow
cnf = DEFunc(CNF(model))
nde = NeuralODE(cnf, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)

# if FM
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note)
#train_fm_model(cnf_model, FM, base, optimizer, batch_size, n_iters, note)

#%%
n_samples = 51
#tspan=torch.linspace(1, 0, 101)
tspan = torch.linspace(0, 1, 201)
nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler", sensitivity="autograd")
#base_samples = rejection_sample(base, n_samples)[:,None]
base_samples = torch.linspace(-2, 2, n_samples)[:,None]
with torch.no_grad():
    #trajectories = cnf_model[1].trajectory(base_samples, t_span=tspan)
    trajectories = nde.trajectory(base_samples, t_span=tspan)
for i in range(50):
    plt.plot(trajectories[:,i], tspan)
plt.xlim([-2,2])

#%%
def compute_log_probs(model, trajectories):
    cnf = DEFunc(CNF(model))
    nde = NeuralODE(cnf, solver="euler", sensitivity="autograd")
    cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)

    lps = torch.zeros(201, n_samples)

    for i, t in enumerate(tqdm(tspan)):
        with torch.no_grad():
            if t > 0:
                aug_traj = (
                    cnf_model[1].trajectory(
                        Augmenter(1, 1)(trajectories[i,...]), t_span=torch.linspace(t, 0, 201),
                        #Augmenter(1, 1)(trajectories[i,...]), t_span=torch.linspace(0, t, 201),
                    )
                )[-1].cpu()
                log_probs = base.log_prob(aug_traj[:, 1]) - aug_traj[:, 0]
            else:
                # for the flow, need t = 1!
                log_probs = base.log_prob(base_samples.squeeze())

            lps[i] = log_probs.squeeze()

    return lps

#%%
lps = compute_log_probs(model, trajectories)

#%%
def plot_trajectories(trajectories, lps, ax):
    probs = lps.exp()
    vmax = base.log_prob(torch.tensor(0)).exp()
    norm = Normalize(vmin=probs.min(), vmax=vmax)

    for i in range(n_samples):
        x = trajectories[:,i].squeeze()
        ax.scatter(x, tspan, s=1, c=probs[:,i], norm=norm, cmap="inferno")

    # Set the limits for the axes
    ax.set_facecolor('black')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)

#%%
# target
#means = torch.tensor([-1.0, 0.5, 1.5])
means = torch.tensor([-1.0, 1.0])
#stds = torch.tensor([0.5, 0.25, 0.25])
stds = torch.tensor([0.5, 0.5])
mixing_probs = torch.ones(means.shape[0])/means.shape[0]
target = MixtureSameFamily(
    Categorical(probs=mixing_probs),
    Normal(loc=means, scale=stds)
)


def plot_pdf(distribution, ax):
    x = torch.linspace(-2, 2, 500)
    probs = distribution.log_prob(x).exp()

    norm = Normalize(vmin=probs.min(), vmax=probs.max())

    ax.scatter(x, probs, c=probs, s=2, norm=norm, cmap="inferno")

    ax.set_facecolor('black')

    ax.set_xlim(x.min(), x.max())
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 0.6)
    ax.set_aspect(2.0)


# %%
fig = plt.figure()
# Create a GridSpec with one row and multiple columns for subplots
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 10/3, 1], hspace=0.0)

# Create subplots with different aspect ratios
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

plot_pdf(target, ax1)
plot_trajectories(trajectories, lps, ax2)
plot_pdf(base, ax3)

plt.tight_layout()

plt.savefig("test.png", dpi=600)
# %%
