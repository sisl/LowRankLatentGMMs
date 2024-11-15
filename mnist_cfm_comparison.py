#%%
import torch
from models import LowRankMixtureModel
import os
import numpy as np

from utils import CropTransform, ReshapeTransform

from torch.distributions import (
    Categorical, MixtureSameFamily, MultivariateNormal
)

from PIL import Image
from utils import samples_to_mosaic

image_shape = [28, 28]          # The input image shape
n_components = 50               # Number of components in the mixture model
n_factors = 5                   # Number of factors - the latent dimension (same for all components)
init_method = 'kmeans'
    
print('Loading pre-trained MFA model...')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = './models/' + 'mnist'
mppca = LowRankMixtureModel(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors).to(device=device)
mppca.load_state_dict(torch.load(os.path.join(model_dir, 'model_c_{}_l_{}.pth'.format(n_components, n_factors, init_method))))
# mppca.sample(10)[0].view(10, 1, 28, 28)
# %%
import os

import matplotlib.pyplot as plt
import torch
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

from torchcfm.optimal_transport import wasserstein

savedir = "models/mnist"
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 128
n_epochs = 3

trans = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=trans, download=True)


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)


#%%
#################################
#            OT-CFM
#################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = 'cpu'
batch_size = 128
n_epochs = 1

sigma = 0.0
model1 = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
optimizer1 = torch.optim.Adam(model1.parameters())
# FM = ConditionalFlowMatcher(sigma=sigma)
FM1 = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node1 = NeuralODE(model1, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)


losses1 = []
distances1 = []
for epoch in range(n_epochs):
    for i, data in enumerate(tqdm(train_loader)):
        optimizer1.zero_grad()
        x1 = data[0].to(device)
        x0 = torch.randn_like(x1)
        #x0 = mppca.sample(batch_size)[0].view(batch_size, 1, 28, 28)
        t, xt, ut = FM1.sample_location_and_conditional_flow(x0, x1)
        vt = model1(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer1.step()

        if (i + 1) % 20 == 0:
            with torch.no_grad():
                #samples = mppca.sample(batch_size)[0].view(batch_size, 1, 28, 28)
                traj = node1.trajectory(
                    torch.randn(100, 1, 28, 28, device=device),
                    t_span=torch.linspace(0, 1, 2, device=device),
                )
                x2 = traj[-1, :]
                w2 = wasserstein(x1, x2)
            losses1.append(loss.item())
            distances1.append(w2)


    node1 = NeuralODE(model1, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        traj = node1.trajectory(
            torch.randn(100, 1, 28, 28, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )

    rnd_samples = traj[-1].view(100,784)
    mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
    image = Image.fromarray((255 * mosaic).astype(np.uint8))
    image.save("normal{}.png".format(epoch))

#%%
#################################
#            OT-CFM
#################################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = 'cpu'
batch_size = 128
n_epochs = 1

sigma = 0.0
model2 = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
optimizer2 = torch.optim.Adam(model2.parameters())
# FM = ConditionalFlowMatcher(sigma=sigma)
FM2 = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node2 = NeuralODE(model2, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)


losses2 = []
distances2 = []
for epoch in range(n_epochs):
    for i, data in enumerate(tqdm(train_loader)):
        optimizer2.zero_grad()
        x1 = data[0].to(device)
        #x0 = torch.randn_like(x1)
        x0 = mppca.sample(batch_size)[0].view(batch_size, 1, 28, 28)
        t, xt, ut = FM2.sample_location_and_conditional_flow(x0, x1)
        vt = model2(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer2.step()

        '''
        if (i + 1) % 20 == 0:
            with torch.no_grad():
                samples = mppca.sample(100)[0].view(100, 1, 28, 28)
                traj = node2.trajectory(
                    samples.to(device),
                    t_span=torch.linspace(0, 1, 2, device=device),
                )
                x2 = traj[-1, :]
                w2 = wasserstein(x1, x2)
            losses2.append(loss.item())
            distances2.append(w2)
        '''
    with torch.no_grad():
        samples = mppca.sample(100)[0].view(100, 1, 28, 28)
        traj = node2.trajectory(
            samples.to(device),
            t_span=torch.linspace(0, 1, 10, device=device),
        )

    rnd_samples = traj[-1].view(100,784)
    mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
    image = Image.fromarray((255 * mosaic).astype(np.uint8))
    image.save("mppca{}.png".format(epoch))

#%%
node1 = NeuralODE(model1, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
with torch.no_grad():
    traj = node1.trajectory(
        torch.randn(100, 1, 28, 28, device=device),
        t_span=torch.linspace(0, 1, 2, device=device),
    )

rnd_samples = traj[-1].view(100,784)
mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
image = Image.fromarray((255 * mosaic).astype(np.uint8))
image.save("normal.png")

#%%
with torch.no_grad():
    samples = mppca.sample(100)[0].view(100, 1, 28, 28)
    traj = node2.trajectory(
        samples.to(device),
        t_span=torch.linspace(0, 1, 10, device=device),
    )

rnd_samples = traj[-1].view(100,784)
mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
image = Image.fromarray((255 * mosaic).astype(np.uint8))
image.save("mppca.png")
# %%
pastelBlue = "#0072B2"
pastelRed = "#F5615C"

plt.figure()
plt.plot(losses1, c=pastelBlue, label="Normal")
plt.plot(losses2, c=pastelRed, label="Mixture")
plt.xlabel("index")
plt.ylabel("L2 Loss")
plt.legend()


plt.figure()
plt.plot(distances1, c=pastelBlue, label="Normal")
plt.plot(distances2, c=pastelRed, label="Mixture")
plt.xlabel("index")
plt.ylabel("L2 Loss")
plt.legend()