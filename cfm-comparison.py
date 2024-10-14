import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

# torchdyn imports
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.models import MLP

from toy_data import generate_data

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"


def autograd_trace(x_out, x_in, **kwargs):
    """
    Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd.
    Code from torchcfm library: https://github.com/atong01/conditional-flow-matching
    """
    trJ = 0.0
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ


class CNF(torch.nn.Module):
    """
    Continuous normalizing flow class. Code from torchcfm library: 
    https://github.com/atong01/conditional-flow-matching
    """
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(True)  # first dimension reserved to divergence propagation
            x_out = self.net(torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1))
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph


def my_plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=1, c='k')
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=pastelRed)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=10, alpha=1, c=pastelRed)
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.gca().set_aspect('equal')
    plt.show()


def compute_avg_log_prob(model, x, device, base):
    # Density plot
    cnf = DEFunc(CNF(model))
    nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
    cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
    with torch.no_grad():
        aug_traj = (
            cnf_model[1]
            .to(device)
            .trajectory(
                Augmenter(1, 1)(x).to(device),
                t_span=torch.linspace(1, 0, 201).to(device),
            )
        )[-1].cpu()
        log_probs = -base.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]

    return log_probs.mean()

#%%
def visualize_model(model, base, title):
    w = 4
    points = 200j
    #points_real = 100
    device = "cpu"
    Y, X = np.mgrid[-w:w:points, -w:w:points]
    gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32)
    points_small = 20j
    points_real_small = 20
    Y_small, X_small = np.mgrid[-w:w:points_small, -w:w:points_small]
    gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
        torch.float32
    )

    torch.manual_seed(42)
    #sample = sample_8gaussians(1024)
    sample = base.sample((1024,))
    ts = torch.linspace(0, 1, 101)

    nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler").to(device)
    # with torch.no_grad():
    traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()

    for i, t in enumerate(ts):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Density plot
        cnf = DEFunc(CNF(model))
        nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
        cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
        with torch.no_grad():
            if t > 0:
                aug_traj = (
                    cnf_model[1]
                    .to(device)
                    .trajectory(
                        Augmenter(1, 1)(gridpoints).to(device),
                        t_span=torch.linspace(t, 0, 201).to(device),    # does this get finer and finer?
                    )
                )[-1].cpu()
                log_probs = base.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
            else:
                log_probs = base.log_prob(gridpoints)
        log_probs = log_probs.reshape(Y.shape)

        ax = axes[0]
        ax.pcolormesh(X, Y, torch.exp(log_probs))

        # Quiver plot
        out = model(
            torch.cat(
                [gridpoints_small, torch.ones((gridpoints_small.shape[0], 1)) * t], dim=1
            ).to(device)
        )
        out = out.reshape([points_real_small, points_real_small, 2]).cpu().detach().numpy()
        ax = axes[1]
        ax.quiver(
            X_small,
            Y_small,
            out[:, :, 0],
            out[:, :, 1],
            np.sqrt(np.sum(out**2, axis=-1)),
            cmap="coolwarm",
            scale=15.0,
            width=0.01,
            pivot="mid",
        )

        # Mapping
        ax = axes[2]
        sample_traj = traj
        ax.scatter(sample_traj[0, :, 0], sample_traj[0, :, 1], s=15, alpha=1, c='k')
        ax.scatter(sample_traj[:i, :, 0], sample_traj[:i, :, 1], s=1, alpha=0.2, c=pastelRed)
        ax.scatter(sample_traj[i, :, 0], sample_traj[i, :, 1], s=15, alpha=1, c=pastelRed)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-w, w)
            ax.set_ylim(-w, w)
            ax.set_aspect('equal')
        plt.tight_layout()

        os.makedirs("figures/trajectory/{}/".format(title), exist_ok=True)
        plt.savefig("figures/trajectory/{}/{:0.2f}.png".format(title, t), dpi=100)
        plt.close()


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
'''
mus = torch.tensor([[-1.0, 3.0], [3.0, 3.0], 
                    [-3.0, 1.0], [1.0, 1.0],
                    [-1.0,-1.0], [3.0,-1.0],
                    [-3.0,-3.0], [1.0,-3.0]])
'''
mus = torch.tensor([[-2.0,0.5],[-0.5,1.5],[-0.5,-1.5],[1.5,1.0],[1.5,-1.0]])
Sigmas = 0.1*torch.eye(2)
#Sigmas = 0.1*torch.eye(2)
pis = torch.ones(mus.shape[0]) / mus.shape[0]
base_mix = MixtureSameFamily(Categorical(pis), MultivariateNormal(mus, Sigmas))
model_mix = MLP(dim=dim, time_varying=True)
optimizer_mix = torch.optim.Adam(model_mix.parameters())
FM_mix = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
#FM_mix = ConditionalFlowMatcher(sigma=sigma)

def train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note):

    savedir = "models/{}-pinwheel".format(note)
    os.makedirs(savedir, exist_ok=True)

    start = time.time()

    losses = []
    nfes = []
    log_probs = []

    for k in range(n_iters):
        optimizer.zero_grad()

        x0 = base.sample((batch_size,))
        x1 = torch.tensor(generate_data("pinwheel", batch_size=batch_size), dtype=torch.float32)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()

        if (k + 1) % 50 == 0:

            end = time.time()
            print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
        
            x1 = torch.tensor(generate_data("pinwheel", batch_size=1024), dtype=torch.float32)
            avg_lp = compute_avg_log_prob(model, x1, 'cpu', base)
            node = NeuralODE(
                torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            )
            with torch.no_grad():
                traj = node.trajectory(
                    base.sample((1024,)),
                    t_span=torch.linspace(0, 1, 100),
                )

            losses.append(loss.item())
            nfes.append(node.vf.nfe)
            log_probs.append(avg_lp)

        if (k + 1) % 500 == 0:
            node = NeuralODE(
                torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            )
            with torch.no_grad():
                traj = node.trajectory(
                    base.sample((1024,)),
                    t_span=torch.linspace(0, 1, 100),
                )
                my_plot_trajectories(traj.cpu().numpy())


    torch.save(model, "{}/otcfm_{}.pt".format(savedir, note))

    return model, losses, nfes, log_probs

# %%
note = "normal"
model, losses, nfes, log_probs = train_fm_model(
    model, FM, base, optimizer, batch_size, n_iters, note)

note_mix = "mixture"
model_mix, losses_mix, nfes_mix, log_probs_mix = train_fm_model(
    model_mix, FM_mix, base_mix, optimizer_mix, batch_size, n_iters, note_mix)


#%%
visualize_model(model, base, "normal")

visualize_model(model_mix, base_mix, "mixture")

#%%
iters = torch.arange(0, n_iters, 100)
plt.figure()
plt.plot(losses, c=pastelBlue)
plt.plot(losses_mix, c=pastelRed)

plt.figure()
plt.plot(nfes, c=pastelBlue)
plt.plot(nfes_mix, c=pastelRed)

plt.figure()
plt.plot(log_probs, c=pastelBlue)
plt.plot(log_probs_mix, c=pastelRed)
# %%
import glob
from PIL import Image

def my_make_gif(frame_folder, out_path, delete_frames=True):
    files = [f for f in glob.glob(f"{frame_folder}/*.png")]
    #print(files)
    files = sorted(files)
    frames = [Image.open(image) for image in files]
    
    frame_one = frames[0]
    frame_one.save(out_path, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

    if delete_frames:
        for f in files:
            os.remove(f)

my_make_gif("figures/trajectory/normal/", "figures/trajectory/normal.gif", delete_frames=False)

my_make_gif("figures/trajectory/mixture/", "figures/trajectory/mixture.gif", delete_frames=False)
# %%
