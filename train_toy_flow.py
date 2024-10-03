#%%
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

pastelBlue = "#0072B2"
pastelRed = "#F5615C"

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)


from construct_transform import create_transform
from toy_data import generate_data
from utils import make_gif, color_base_dist

from nflows.distributions.base import Distribution

class MixtureModel(Distribution):

    def __init__(self, mus, Sigmas, pis):
        super().__init__()
        self.mus = mus
        self.Sigmas = Sigmas
        self.pis = pis

        self.distribution = MixtureSameFamily(
            Categorical(pis),
            MultivariateNormal(mus, Sigmas)
        )

    def _log_prob(self, inputs, context):
        return self.distribution.log_prob(inputs)

    def _sample(self, num_samples, context):
        return self.distribution.sample((num_samples,))

    def _mean(self, context):
        return self.mus

#%%
torch.manual_seed(0)
np.random.seed(0)

args = argparse.Namespace()

args.num_flow_steps = 4
args.features = 2
args.hidden_features = 256
args.context_features = None
args.linear = 'lu'
args.base = 'rq-c'
args.num_bins = 8
args.tail_bound = 4.0
args.use_batch_norm = False
args.target = 'pinwheel'
args.batch_size = 256
args.grad_norm_clip_value = None

use_cuda = torch.cuda.is_available()
device = torch.device(args.device if use_cuda else "cpu")

'''
mus = torch.tensor([[-1.0, 3.0], [3.0, 3.0], 
                    [-3.0, 1.0], [1.0, 1.0],
                    [-1.0,-1.0], [3.0,-1.0],
                    [-3.0,-3.0], [1.0,-3.0]])
'''
mus = torch.tensor([[-2.0,0.5],[-0.5,1.5],[-0.5,-1.5],[1.5,1.0],[1.5,-1.0]])
Sigmas = 0.1*torch.eye(2)
pis = torch.ones(mus.shape[0]) / mus.shape[0]
mixture_base = MixtureModel(mus, Sigmas, pis)
mixture_flow = Flow(create_transform(args), mixture_base)
mixture_optimizer=torch.optim.Adam(mixture_flow.parameters(), lr=1e-3)

standard_base = StandardNormal(shape=[args.features])
standard_flow = Flow(create_transform(args), standard_base)
standard_optimizer=torch.optim.Adam(standard_flow.parameters(), lr=1e-3)


z = standard_flow.sample(1000)
with torch.no_grad():
    plt.scatter(z[:,0], z[:,1])

z = mixture_flow.sample(1000)
with torch.no_grad():
    plt.scatter(z[:,0], z[:,1])

plt.gca().set_aspect('equal')
#%%
num_iter = 5000
def train_flow(flow, num_iter, args, optimizer, device, note):

    flow.to(device)

    iters = []
    losses = []
    condition_numbers = []

    for i in range(num_iter):
        flow.train()
        # Create target data
        x = torch.tensor(
                generate_data(args.target,batch_size=args.batch_size
            ), dtype=torch.float32)
        x = x.to(device)

        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()

        loss.backward()
        if args.grad_norm_clip_value is not None:
            clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
        optimizer.step()

        if (i+1) % 10 == 0:
            flow.eval()
            print("Iteration {}".format(i + 1))
            print("Loss {:.8f}".format(loss.item()))
            losses.append(loss.item())
            iters.append(i)

            # do stuff here
            u = flow._distribution.sample(100)
            
            jacobian = torch.autograd.functional.jacobian(flow._transform.inverse, u)[0]
            jacobian = jacobian.sum(dim=2)
            svd = np.linalg.svd(jacobian, compute_uv=False)
            cond_number = np.max(svd) / np.min(svd)
            condition_numbers.append(cond_number)

        if i % 100 == 0:
            flow.eval()
            xline=torch.linspace(-4, 4, 200)
            yline = torch.linspace(-4, 4, 200)
            xgrid, ygrid = torch.meshgrid(xline, yline, indexing = 'ij')
            xyinput = torch.cat([xgrid.reshape(-1,1), ygrid.reshape(-1,1)], dim=1)
            xyinput = xyinput.to(device)
            with torch.no_grad():
                zgrid=flow.log_prob(xyinput).exp().reshape(200,200)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(), levels=100, cmap='inferno')
            plt.xlim([-4,4])
            plt.ylim([-4,4])
            plt.gca().set_aspect(True)
            plt.title("iteration {}".format(i+1))
            plt.tight_layout()
            plt.savefig("figures/{}{:04d}.png".format(note, int((i+1)/10)))

    make_gif("figures/", "figures/{}.gif".format(note), delete_frames=True)

    flow.to('cpu')

    return losses, condition_numbers, iters


mixture_losses, mixture_condition_numbers, iters = \
    train_flow(mixture_flow, num_iter, args, mixture_optimizer, device, "mixture")

standard_losses, standard_condition_numbers, iters = \
    train_flow(standard_flow, num_iter, args, standard_optimizer, device, "standard")

us = standard_base.sample(1000)
zs, logabsdet_s = standard_flow._transform.inverse(us)

um = mixture_base.sample(1000)
zm, logabsdet_m = mixture_flow._transform.inverse(um)

plt.figure()
plt.plot(iters, standard_losses, c=pastelRed, label="Standard") 
plt.plot(iters, mixture_losses, c=pastelBlue, label="Mixture")
plt.title("Training Loss")
plt.xlabel("iteration")
plt.ylabel("NLL")
plt.legend()
plt.savefig("figures/loss.png")

plt.figure()
plt.plot(iters, standard_condition_numbers, c=pastelRed, label="Standard") 
plt.plot(iters, mixture_condition_numbers, c=pastelBlue, label="Mixture")
plt.title("Condition Number")
plt.xlabel("iteration")
plt.ylabel("k(J)")
plt.ylim([0, 500])
plt.legend()
plt.savefig("figures/condition.png")

plt.figure()
with torch.no_grad():
    plt.hist(logabsdet_s, color=pastelRed, bins=20, alpha=0.8, label="Standard")
    plt.hist(logabsdet_m, color=pastelBlue, bins=20, alpha=0.8, label="Mixture")
    plt.title("Log of Jacobian Determinant")
    plt.xlabel(r"$\log |\det ( J )|$")
    plt.legend()
plt.savefig("figures/logabsdet.png")

#%%

z = mixture_flow.sample(1000)
with torch.no_grad():
    plt.scatter(z[:,0], z[:,1])


# %%

def plot_vector(flow, n_samples):
    base_dist = flow._distribution
    u = base_dist.sample(n_samples)
    colors = color_base_dist(u)
    z = u.clone()

    rev_inv_transforms = \
        [transform.inverse for transform in flow._transform._transforms[::-1]]

    plt.figure(figsize=(8,8))
    plt.scatter(u[:,0].numpy(), u[:,1].numpy(),s=10, c=colors)
    plt.savefig("iter{}.png".format(i), dpi=600)
    plt.gca().set_xlim(-4,4)
    plt.gca().set_ylim(-4,4)

    for i, inv_transform in enumerate(rev_inv_transforms):
        z_new, logabsdet = inv_transform(z)

        dx = z_new[:, 0] - z[:, 0]
        dy = z_new[:, 1] - z[:, 1]

        with torch.no_grad():
            plt.quiver(z[:, 0], z[:, 1], dx, dy, angles='xy', scale_units='xy', scale=1, color=colors)
        z = z_new

    with torch.no_grad():
        plt.scatter(z[:,0].numpy(), z[:,1].numpy(),s=10, c=colors)


#%%
x = torch.tensor(
        generate_data("pinwheel", batch_size=1000), dtype=torch.float32)
plt.scatter(x[:,0], x[:,1])


mu1 = torch.tensor([-0.5, 1.5])
cov1 = torch.tensor([[0.1, 0.0], [0.0, 0.1]])

dist1 = MultivariateNormal(mu1, cov1)

x1 = dist1.sample((300,))

plt.scatter(x1[:,0], x1[:,1])
# %%



mus = torch.tensor([[-2.0,0.5],[-0.5,1.5],[-0.5,-1.5],[1.5,1.0],[1.5,-1.0]])
Sigmas = 0.1*torch.eye(2)
pis = torch.ones(mus.shape[0]) / mus.shape[0]
mixture_base = MixtureModel(mus, Sigmas, pis)

standard_base = StandardNormal([2])

xline=torch.linspace(-4, 4, 200)
yline = torch.linspace(-4, 4, 200)
xgrid, ygrid = torch.meshgrid(xline, yline, indexing = 'ij')
xyinput = torch.cat([xgrid.reshape(-1,1), ygrid.reshape(-1,1)], dim=1)

with torch.no_grad():
    zgrid=standard_base.log_prob(xyinput).exp().reshape(200,200)

plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(), levels=100, cmap='inferno')

x = torch.tensor(
        generate_data("pinwheel", batch_size=1000), dtype=torch.float32)
plt.scatter(x[:,0], x[:,1], s=10, c=pastelBlue)

plt.xlim([-4,4])
plt.ylim([-4,4])
plt.gca().set_aspect(True)
plt.tight_layout()
# %%
