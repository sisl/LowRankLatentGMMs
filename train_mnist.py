import argparse
import copy
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# file imports
from models import LowRankMixtureModel
from utils import samples_to_mosaic, sample_base

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='normal',
                    choices=['normal', 'mppca'])
parser.add_argument('--model_file', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1)
args = parser.parse_args()

# image shape [H, W, n_channels]
image_shape = [28, 28, 1]
n_features = np.prod(image_shape)

model_dir = './models/mnist/'
figure_dir = './figures/mnist/flow_{}/'.format(args.base)
os.makedirs(figure_dir, exist_ok=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define base distribution
if args.base == "mppca":
    print('Loading pre-trained MPPCA model...')
    model_dict = torch.load(model_dir + args.model_file, weights_only=True)
    n_components, n_features, n_factors = model_dict['W'].shape
    base = LowRankMixtureModel(
        n_components=n_components,
        n_features=n_features,
        n_factors=n_factors
    )
    base.load_state_dict(model_dict)
    base.to(device)
else:
    base = MultivariateNormal(
        torch.zeros(n_features).to(device), 
        torch.eye(n_features).to(device)
    )

# main training loop
def train():
    # read in data
    trans = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))    
        ]
    )
    train_set = MNIST(root='./data', train=True, transform=trans, download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # define NODE model
    net_model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)

    # define flow-matching model
    sigma = 0.0
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    
    # define optimizer
    optimizer = torch.optim.Adam(net_model.parameters(), lr=2e-4)

    losses = []
    counter = 0
    for epoch in range(args.n_epochs):
        for i, data in enumerate(tqdm(train_loader, desc="epoch {}: ".format(epoch+1))):
            optimizer.zero_grad()
            x1 = data[0].to(device)
            x0 = sample_base(base, args.batch_size, image_shape, True)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            if counter % 20 == 0:
                losses.append(loss.item())
            counter += 1

        # save image
        net_model.eval()
        model_ = copy.deepcopy(net_model)
        node_ = NeuralODE(model_, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            samples = sample_base(base=base, N=100, image_shape=image_shape, with_noise=False)
            traj = node_.trajectory(
                samples.to(device),
                t_span=torch.linspace(0, 1, 2, device=device),
            )
            print(node_.vf.nfe)
        net_model.train()
        rnd_samples = traj[-1].view(100,784)
        mosaic = samples_to_mosaic(rnd_samples, image_shape=[28,28])
        image = Image.fromarray((255 * mosaic).astype(np.uint8))
        image.save(figure_dir + "{}_base_epoch_{}.png".format(args.base, epoch))


if __name__ == "__main__":
    train()
