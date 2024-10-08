#%%
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

from toy_data import generate_data

from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)

pastelBlue = "#0072B2"
pastelRed = "#F5615C"

note = "normal"

savedir = "models/{}-checkerboard".format(note)
os.makedirs(savedir, exist_ok=True)

def my_plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=1, c='k')
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=pastelRed)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c=pastelRed)
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

#%%
sigma = 0.1
dim = 2
batch_size = 256
n_iters = 20000


# standard base model
base = MultivariateNormal(torch.zeros(dim), torch.eye(2))
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
#FM = ConditionalFlowMatcher(sigma=sigma)

# mixture base model
mus = torch.tensor([[-1.0, 3.0], [3.0, 3.0], 
                    [-3.0, 1.0], [1.0, 1.0],
                    [-1.0,-1.0], [3.0,-1.0],
                    [-3.0,-3.0], [1.0,-3.0]])
Sigmas = 0.1*torch.eye(dim)
pis = torch.ones(mus.shape[0]) / mus.shape[0]
base_mix = MixtureSameFamily(Categorical(pis), MultivariateNormal(mus, Sigmas))
model_mix = MLP(dim=dim, time_varying=True)
optimizer_mix = torch.optim.Adam(model_mix.parameters())
FM_mix = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
#FM_mix = ConditionalFlowMatcher(sigma=sigma)

def train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note):
    start = time.time()

    losses = []
    nfes = []

    for k in range(n_iters):
        optimizer.zero_grad()

        x0 = base.sample((batch_size,))
        x1 = torch.tensor(generate_data("checkerboard", batch_size), dtype=torch.float32)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

        if (k + 1) % 100 == 0:

            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
        
            node = NeuralODE(
                torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            )
            with torch.no_grad():
                traj = node.trajectory(
                    base.sample((1024,)),
                    t_span=torch.linspace(0, 1, 100),
                )
                my_plot_trajectories(traj.cpu().numpy())

            losses.append(loss.item())
            nfes.append(node.vf.nfe)

    torch.save(model, "{}/otcfm_{}.pt".format(savedir, note))

    return model, losses, nfes

# %%
model, losses, nfes = train_fm_model(
    model, FM, base, optimizer, batch_size, n_iters, "standard")

#model_mix, losses_mix, nfes_mix = train_fm_model(
#    model_mix, FM_mix, base_mix, optimizer_mix, batch_size, n_iters, "mixture")

'''
plt.figure()
plt.plot(losses, c=pastelBlue)
plt.plot(losses_mix, c=pastelRed)

plt.figure()
plt.plot(nfes, c=pastelBlue)
plt.plot(nfes_mix, c=pastelRed)
'''

# %%
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter

def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.0
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[
            0
        ][:, i]
    return trJ

class CNF(torch.nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(
                True
            )  # first dimension reserved to divergence propagation
            # the neural network will handle the data-dynamics here
            x_out = self.net(
                torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1)
            )
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
    

cnf = DEFunc(CNF(model))
nde = NeuralODE(cnf, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)

w = 4
points = 100j
points_real = 100
device = "cpu"
Y, X = np.mgrid[-w:w:points, -w:w:points]
gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32)

t = 1
with torch.no_grad():
    if t > 0:
        aug_traj = (
            cnf_model[1]
            .to(device)
            .trajectory(
                Augmenter(1, 1)(gridpoints).to(device),
                t_span=torch.linspace(t, 0, 201).to(device),
            )
        )[-1].cpu()
        log_probs = base.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
    else:
        log_probs = base.log_prob(gridpoints)
        
plt.figure()
ax = plt.gca()
log_probs = log_probs.reshape(Y.shape)
ax.pcolormesh(X, Y, torch.exp(log_probs), vmax=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)
#ax.set_title(f"{name}", fontsize=30)
# %%
points_small = 20j
points_real_small = 20
Y_small, X_small = np.mgrid[-w:w:points_small, -w:w:points_small]
gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
    torch.float32
)

out = model(
    torch.cat(
        [gridpoints_small, torch.ones((gridpoints_small.shape[0], 1)) * t], dim=1
    ).to(device)
)
out = out.reshape([points_real_small, points_real_small, 2]).cpu().detach().numpy()

plt.figure()
ax = plt.gca()
ax.quiver(
    X_small,
    Y_small,
    out[:, :, 0],
    out[:, :, 1],
    np.sqrt(np.sum(out**2, axis=-1)),
    cmap="coolwarm",
    scale=50.0,
    width=0.015,
    pivot="mid",
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-w, w)

#%%
i = 100
ts = torch.linspace(0, 1, 101)

sample = base.sample((1024,))
nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler").to(device)
# with torch.no_grad():
sample_traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()

plt.figure()
ax = plt.gca()

ax.scatter(sample_traj[0, :, 0], sample_traj[0, :, 1], s=10, alpha=1, c='k')
ax.scatter(sample_traj[:i, :, 0], sample_traj[:i, :, 1], s=0.2,alpha=0.2, c=pastelRed)
ax.scatter(sample_traj[i, :, 0], sample_traj[i, :, 1], s=6, alpha=1, c=pastelRed)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)

