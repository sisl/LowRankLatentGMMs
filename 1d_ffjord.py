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
from utils import plot_1d_trajectories, plot_pdf


sigma = 0.01
dim = 1
batch_size = 128
n_iters = 200


# base
base = Normal(loc=0.0, scale=1.0)
# target
means = torch.tensor([-1.0, 1.0])

stds = torch.tensor([0.5, 0.5])
mixing_probs = torch.ones(means.shape[0])/means.shape[0]
target = MixtureSameFamily(
    Categorical(probs=mixing_probs),
    Normal(loc=means, scale=stds)
)


model = MLP(dim=dim, w=32, time_varying=True)

def train_cnf_model(model, base, optimizer, batch_size, n_iters):
    start = time.time()
    for k in range(n_iters):
        optimizer.zero_grad()
        x1 = target.sample((batch_size,))[:,None]
        t, xtrJ = model(x1)
        logprob = base.log_prob(xtrJ[1, :,1:]) - xtrJ[1, :,0]
        loss = -torch.mean(logprob)
        loss.backward()
        optimizer.step()

        if (k + 1) % 20 == 0:
            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end


cnf = DEFunc(CNF(model))
nde = NeuralODE(cnf, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)

# if FM
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
train_cnf_model(cnf_model, base, optimizer, batch_size, n_iters)

#%%
n_samples = 51
tspan=torch.linspace(1, 0, 201)
#tspan = torch.linspace(0, 1, 201)
nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler", sensitivity="autograd")
#base_samples = rejection_sample(base, n_samples)[:,None]
base_samples = torch.linspace(-2, 2, n_samples)[:,None]
with torch.no_grad():
    #trajectories = cnf_model[1].trajectory(base_samples, t_span=tspan)
    trajectories = nde.trajectory(base_samples, t_span=tspan)
for i in range(50):
    plt.plot(trajectories[:,i], tspan.flip(-1))
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
                        #Augmenter(1, 1)(trajectories[i,...]), t_span=torch.linspace(t, 0, 201),
                        Augmenter(1, 1)(trajectories[-i,...]), t_span=torch.linspace(0, t, 201),
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


# %%
fig = plt.figure()
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 5/3, 1], hspace=0.0)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

plot_pdf(target, ax1)
plot_1d_trajectories(trajectories, base, lps, tspan, ax2)
plot_pdf(base, ax3)

plt.tight_layout()

plt.savefig("figures/vanilla_cnf_1d.png", dpi=600)