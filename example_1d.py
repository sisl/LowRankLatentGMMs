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
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import MLP
from torchcfm.utils import torch_wrapper

from cnf import compute_log_probs
from toy_data import generate_data
from utils import make_gif, plot_trajectories, visualize_model

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"

#%%
sigma = 0.1
dim = 1
batch_size = 64
n_iters = 1000




# standard base model
base = MultivariateNormal(torch.zeros(dim), torch.eye(1))
model = MLP(dim=dim, w=16, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

def train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note):

    savedir = "models/{}-1d".format(note)
    os.makedirs(savedir, exist_ok=True)

    start = time.time()

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
        
    return model

# %%
note = "normal-FM"
model = train_fm_model(model, FM, base, optimizer, batch_size, n_iters, note)



#%%
dim = 1
base = MultivariateNormal(torch.zeros(dim), torch.eye(1))


# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import viridis
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# Generate values between -3 and 3
x = torch.linspace(-2, 2, 500)
y = base.log_prob(x[:,None]).exp()

# Normalize y values to [0, 1] for colormap
norm = Normalize(vmin=y.min(), vmax=y.max())

# Prepare the points for line segments
points = np.array([x.numpy(), y.numpy()]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a line collection for the plot, mapping y-values to viridis colormap
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(y.numpy())
lc.set_linewidth(3)

# Create the plot
fig, ax = plt.subplots()

# Add the line collection to the axis
ax.add_collection(lc)

# Set the limits for the axes
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 1)
plt.gca().set_aspect(2.0)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Standard Normal Distribution PDF (Color by Y-Value)')

# Show the color bar
#plt.colorbar(lc, ax=ax)

# Show the plot
plt.show()

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import viridis
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Define the PDF of a Gaussian
def gaussian_pdf(x, mean, std):
    return torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * torch.sqrt(2 * torch.pi))

# Define the PDF of a Gaussian Mixture Model (GMM)
def gmm_pdf(x, means, stds, weights):
    pdf = torch.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        pdf += weight * gaussian_pdf(x, mean, std)
    return pdf

# Parameters for the GMM with three modes
means = torch.tensor([-2.0, 0.0, 2.0])   # Means of the three Gaussians
stds = torch.tensor([0.5, 1.0, 0.75])    # Standard deviations of the three Gaussians
weights = torch.tensor([0.3, 0.4, 0.3])  # Weights of the three Gaussians

# Generate values between -5 and 5
x = torch.linspace(-5, 5, 500)
y = gmm_pdf(x, means, stds, weights)

# Normalize y values to [0, 1] for colormap
norm = Normalize(vmin=y.min(), vmax=y.max())

# Prepare the points for line segments
points = np.array([x.numpy(), y.numpy()]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a line collection for the plot, mapping y-values to viridis colormap
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(y.numpy())
lc.set_linewidth(2)

# Create the plot
fig, ax = plt.subplots()

# Add the line collection to the axis
ax.add_collection(lc)

# Set the limits for the axes
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('PDF')
ax.set_title('Gaussian Mixture Model PDF (3 Modes, Color by Y-Value)')

# Show the color bar
plt.colorbar(lc, ax=ax)

# Show the plot
plt.show()

# %%
