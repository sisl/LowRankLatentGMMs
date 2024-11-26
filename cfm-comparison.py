#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

# torchdyn imports
from torchdyn.core import DEFunc, NeuralODE

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.models import MLP
from torchcfm.utils import torch_wrapper

from cnf import compute_log_probs
from utils import make_gif, plot_trajectories, visualize_model, generate_data

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"

#%%
sigma = 0.1
dim = 2
batch_size = 256
n_iters = 1000


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
            avg_lp = compute_log_probs(model, x1, 1, 'cpu', base)

            losses.append(loss.item())
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
                plot_trajectories(traj.cpu().numpy())


    torch.save(model, "{}/otcfm_{}.pt".format(savedir, note))

    return model, losses, log_probs

# %%
note = "normal"
model, losses, log_probs = train_fm_model(
    model, FM, base, optimizer, batch_size, n_iters, note)

note_mix = "mixture"
model_mix, losses_mix, log_probs_mix = train_fm_model(
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
plt.plot(log_probs, c=pastelBlue)
plt.plot(log_probs_mix, c=pastelRed)

make_gif("figures/trajectory/normal/", "figures/trajectory/normal.gif", delete_frames=False)

make_gif("figures/trajectory/mixture/", "figures/trajectory/mixture.gif", delete_frames=False)