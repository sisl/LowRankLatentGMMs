# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import argparse
import copy
import os
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchvision.datasets import FGVCAircraft
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import trange

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

# file imports
from models import LowRankMixtureModel
from utils import infiniteloop, sample_base

#args = argparse.Namespace()
#args.base = "mppca"
parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='normal',
                    choices=['normal', 'mppca'])
args = parser.parse_args()

args.model_file = "model_c_300_l_5_init_rnd_samples.pth"
args.num_channel = 64
args.lr = 2e-4
args.grad_clip = 1.0
args.total_steps = 100000
args.warmup = 500
args.batch_size = 64
args.save_step = 1000

# image shape [H, W, n_channels]
image_shape = [64, 64, 3]
n_features = np.prod(image_shape)

model_dir = './models/fgvc-aircraft/'
figure_dir = './figures/fgvc-aircraft/flow_{}/'.format(args.base)
os.makedirs(figure_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# define warmup learning rate
def warmup_lr(step):
    return min(step, args.warmup) / args.warmup

# main training loop
def train():
    # read in dataset
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    trans = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    dataset = FGVCAircraft(root='./data', split = 'trainval', transform=trans, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # Define NODE model
    net_model = UNetModelWrapper(
        dim=(3, 64, 64),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 3, 4],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    # show NODE model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # define flow-matching model
    sigma = 0.0
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    # define optimizer and scheduler
    optimizer = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = sample_base(base, args.batch_size, image_shape, False)
            x0 = x0.to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            clip_grad_norm_(net_model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # sample and Saving the weights
            if args.save_step > 0 and step % args.save_step == 0:
                net_model.eval()
                model_ = copy.deepcopy(net_model)
                node_ = NeuralODE(model_, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
                with torch.no_grad():
                    torch.manual_seed(0)
                    samples = sample_base(base=base, N=64, image_shape=image_shape, with_noise=False)
                    traj = node_.trajectory(
                        samples.to(device),
                        t_span=torch.linspace(0, 1, 2, device=device),
                    )
                    #print(node_.vf.nfe)
                net_model.train()
                traj = traj[-1, :].view([-1, 3, 64, 64])
                traj = traj * std[:, None, None].to(device) + mean[:, None, None].to(device)
                save_image(traj, figure_dir + f"{args.base}_base_step_{step}.png", nrow=8)


if __name__ == "__main__":
    train()