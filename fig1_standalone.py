#%%
#*******************************************************************************
# imports and setup
#*******************************************************************************
# packages
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE

# latex rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for all text rendering
    "font.family": "serif",  # Use serif font (default LaTeX font)
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}"  # Load extra packages
})

# torchcfm imports
from torchcfm.conditional_flow_matching import (
    VariancePreservingConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchcfm.models.models import MLP

# file imports
from models.cnf import torch_wrapper
from models.mppca import MPPCA
from utils.utils import generate_data

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"

# define the grid
n_steps = 400
x = torch.linspace(-4, 4, n_steps)
y = torch.linspace(-4, 4, n_steps)
xgrid, ygrid = torch.meshgrid(x, y, indexing = 'ij')
xyinput = torch.cat([xgrid.reshape(-1,1), ygrid.reshape(-1,1)], dim=1)
# convert to numpy for plotting
x_np, y_np, = xgrid.numpy(), ygrid.numpy()

# model parameters
n_features = 2
n_components = 4
n_factors = 1
sigma = 0.2

# define base distributions
normal_base = MultivariateNormal(torch.zeros(n_features), torch.eye(n_features))

print("\n****************************************")
print("Fitting MPPCA base.")
print("****************************************\n")
mppca_base = MPPCA(n_components=n_components, n_features=n_features, n_factors=n_factors)
dataset = torch.tensor(generate_data("moons", batch_size=20000), dtype=torch.float32)
fit_ll = mppca_base.fit(dataset, max_iterations=20)

normal_prior = normal_base.log_prob(xyinput).exp().reshape(n_steps,n_steps).numpy()
mppca_prior = mppca_base.log_prob(xyinput).exp().reshape(n_steps,n_steps).numpy()

# training parameters
batch_size = 256
n_iters = 10000

# plotting parameters
t_steps = 400
n_plot = 100
n_target = 2048
shift = 0.0
# increments to shift trajectories
increments = torch.arange(1, t_steps + 1).view(t_steps, 1) * (shift/t_steps)
# generate samples from the target density
target = torch.tensor(generate_data("moons", batch_size=n_target), dtype=torch.float32)
target[:,0] = target[:,0] + shift

# create the figure
#fig, axs = plt.subplots(1,2, figsize=(6,8), constrained_layout=True)
fig, axs = plt.subplots(1,3, figsize=(14,5), constrained_layout=True)

#*******************************************************************************
# VPCFM (Normal base)
#*******************************************************************************
torch.manual_seed(42)

model = MLP(dim=n_features, time_varying=True)
optimizer = torch.optim.SGD(model.parameters())
FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)

print("\n****************************************")
print("Training VP-CFM, normal base.")
print("****************************************\n")
start = time.time()
for k in range(n_iters):
    optimizer.zero_grad()

    x0 = normal_base.sample((batch_size,))
    x1 = torch.tensor(generate_data("moons", batch_size=batch_size), dtype=torch.float32)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 200 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")

torch.manual_seed(42)
node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

with torch.no_grad():
    traj = node.trajectory(normal_base.sample((n_plot,)), t_span=torch.linspace(0, 1, t_steps))

traj[..., 0] = traj[..., 0] + increments

axs[0].set_title(r"\textbf{CFM with Normal base}", fontsize=16)
axs[0].contour(x_np, y_np, normal_prior, levels=5, cmap='viridis')
axs[0].scatter(target[:, 0], target[:, 1], s=20, alpha=0.15, c='k', edgecolors='none')
axs[0].scatter(traj[:, :n_plot, 0], traj[:, :n_plot, 1], s=0.2, alpha=0.2, c=pastelRed)
axs[0].scatter(traj[-1, :n_plot, 0], traj[-1, :n_plot, 1], s=10, alpha=1, c=pastelRed)
axs[0].scatter(traj[0, :n_plot, 0], traj[0, :n_plot, 1], s=10, alpha=1, c='k')
axs[0].set_xlim([-3.85, 3.85])
axs[0].set_ylim([-2.5, 3])
axs[0].set_aspect('equal')
axs[0].set_axis_off()

#*******************************************************************************
# OTCFM (Normal base)
#*******************************************************************************
torch.manual_seed(42)

model = MLP(dim=n_features, time_varying=True)
optimizer = torch.optim.SGD(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

print("\n****************************************")
print("Training OT-CFM, normal base.")
print("****************************************\n")
start = time.time()
for k in range(n_iters):
    optimizer.zero_grad()

    x0 = normal_base.sample((batch_size,))
    x1 = torch.tensor(generate_data("moons", batch_size=batch_size), dtype=torch.float32)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 200 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")


torch.manual_seed(42)
node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

with torch.no_grad():
    traj = node.trajectory(normal_base.sample((n_plot,)), t_span=torch.linspace(0, 1, t_steps))

traj[..., 0] = traj[..., 0] + increments

axs[1].set_title(r"\textbf{OT-CFM with Normal base}", fontsize=16)
axs[1].contour(x_np, y_np, normal_prior, levels=5, cmap='viridis')
axs[1].scatter(target[:, 0], target[:, 1], s=20, alpha=0.15, c='k', edgecolors='none')
axs[1].scatter(traj[:, :n_plot, 0], traj[:, :n_plot, 1], s=0.2, alpha=0.2, c=pastelRed)
axs[1].scatter(traj[-1, :n_plot, 0], traj[-1, :n_plot, 1], s=10, alpha=1, c=pastelRed)
axs[1].scatter(traj[0, :n_plot, 0], traj[0, :n_plot, 1], s=10, alpha=1, c='k')
axs[1].set_xlim([-3.85, 3.85])
axs[1].set_ylim([-2.5, 3])
axs[1].set_aspect('equal')
axs[1].set_axis_off()

#*******************************************************************************
# OTCFM (MPPCA base)
#*******************************************************************************
torch.manual_seed(42)

model = MLP(dim=n_features, time_varying=True)
optimizer = torch.optim.SGD(model.parameters())
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

print("\n****************************************")
print("Training OT-CFM, MPPCA base.")
print("****************************************\n")
start = time.time()
for k in range(n_iters):
    optimizer.zero_grad()

    x0 = mppca_base.sample(batch_size, with_noise=True)[0]
    x1 = torch.tensor(generate_data("moons", batch_size=batch_size), dtype=torch.float32)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    loss.backward()
    optimizer.step()

    if (k + 1) % 200 == 0:
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")


torch.manual_seed(42)
node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

with torch.no_grad():
    traj = node.trajectory(mppca_base.sample(n_plot, with_noise=True)[0], t_span=torch.linspace(0, 1, t_steps))

traj[..., 0] = traj[..., 0] + increments

axs[2].set_title(r"\textbf{OT-CFM with MPPCA base}", fontsize=16)
axs[2].contour(x_np, y_np, mppca_prior, levels=5, cmap='viridis')
axs[2].scatter(target[:, 0], target[:, 1], s=20, alpha=0.15, c='k', edgecolors='none')
axs[2].scatter(traj[:, :n_plot, 0], traj[:, :n_plot, 1], s=0.2, alpha=0.2, c=pastelRed)
axs[2].scatter(traj[-1, :n_plot, 0], traj[-1, :n_plot, 1], s=10, alpha=1, c=pastelRed)
axs[2].scatter(traj[0, :n_plot, 0], traj[0, :n_plot, 1], s=10, alpha=1, c='k')
axs[2].set_xlim([-3.85, 3.85])
axs[2].set_ylim([-2.5, 3])
axs[2].set_aspect('equal')
axs[2].set_axis_off()

# save figure
plt.savefig("fig1.png", dpi=600, bbox_inches='tight')
