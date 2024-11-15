#%%
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from PIL import Image
import time
from torchvision import datasets, transforms

# torchcfm imports
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# torchdyn imports
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter

# file imports
from models import LowRankMixtureModel
from utils import samples_to_mosaic
from cnf import CNF

'''
parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, default='normal',
                    choices=['normal', 'mppca'])
parser.add_argument('--model_file', type=str)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1)
args = parser.parse_args()
'''

args = argparse.Namespace()
args.base = "mppca"
args.model_file = "model_c_50_l_5.pth"
args.batch_size = 128
args.n_epochs = 1


image_shape = [28, 28]
n_features = np.prod(image_shape)

model_dir = './models/mnist/'
figure_dir = './figures/mnist/flow_{}/'.format(args.base)

# read in data
trans = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = datasets.MNIST(root='./data', train=False, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, drop_last=True
)

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

# define flow-matching model
sigma = 0.0
model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

optimizer = torch.optim.Adam(model.parameters())

def sample_base(base, N, image_shape, device):
    if type(base) == LowRankMixtureModel:
        samples = base.sample(N)[0].view(N, 1, image_shape[0], image_shape[1])
    else:
        samples = base.sample((N,)).view(N, 1, image_shape[0], image_shape[1])
        
    return samples.to(device)

#%%
'''
# main training loop
start = time.time()
for epoch in range(args.n_epochs):
    for i, data in enumerate(tqdm(train_loader, desc="epoch {}: ".format(epoch+1))):
        optimizer.zero_grad()
        x1 = data[0].to(device)
        x0 = sample_base(base, args.batch_size, image_shape, device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        samples = sample_base(base, 100, image_shape, device)
        traj = node.trajectory(
            samples,
            t_span=torch.linspace(0, 1, 2, device=device),
        )
        print(node.vf.nfe)

    rnd_samples = traj[-1].view(100,784)
    mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
    image = Image.fromarray((255 * mosaic).astype(np.uint8))
    image.save(figure_dir + "{}_base_epoch_{}.png".format(args.base, epoch))
'''
# %%


images = []
for img, label in test_set:
    images.append(img)
# stack images and labels into tensor format
images_tensor = torch.stack(images)

my_test = images_tensor[:200, ...]

#%%

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd.
    Code from torchdyn library: https://github.com/DiffEqML/torchdyn
    """
    x_out = x_out.reshape(x_out.shape[0], -1)
    x_in = x_in.reshape(x_in.shape[0], -1)
    noise = torch.randn_like(x_in)
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True, allow_unused=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ

class MYCNF(torch.nn.Module):
    """
    Continuous normalizing flow class.
    Code from torchdyn library: https://github.com/DiffEqML/torchdyn
    """
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else hutch_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(True)  # first dimension reserved to divergence propagation
            #x_out = self.net(torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1))
            x_out = self.net(t = t * torch.ones(x.shape[0], 1).type_as(x_in).to(device), x=x_in)#x=torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1))
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
    

cnf = DEFunc(MYCNF(model))
nde = NeuralODE(cnf, solver="euler", sensitivity="autograd")
cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)


t = 1
#flattened = my_test.reshape(200, 784)
with torch.no_grad():
    aug_traj = (
        cnf_model[1].to(device).trajectory(
            Augmenter(1, 1)(my_test).to(device), t_span=torch.linspace(t, 0, 101).to(device),
        )
    )[-1].cpu()
    log_probs = base.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
# %%
