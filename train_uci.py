#%%
import argparse
import os
import time
import torch
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import LowRankMixtureModel
from mmd import mmd

from cnf import cnf_test_metrics, autograd_trace

from uci_dataset import UCIDataset, create_data_loaders, load_config
from early_stopping import EarlyStopping

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchcfm.models.models import MLP

#parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='power',
#                    choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'])
#args = parser.parse_args()
#dataset_name = args.dataset
sigma = 0.1
epochs = 100

flow = 'ot-fm' # 'fm'
base = 'mppca' # 'normal'
dataset = 'power'
data_dir = 'data/UCI/'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# create model directory
model_dir = './models/{}/{}'.format(dataset, flow + '-' + base)
os.makedirs(model_dir, exist_ok=True)

w = None

# read in dataset and associated hyperparameters
data_path = os.path.join(data_dir, dataset + '.npy')
hyperparameters = load_config('experiments.json', dataset)

n_features = hyperparameters['n_features']
n_components = hyperparameters['n_components']
n_factors = hyperparameters['n_factors']
em_iters = hyperparameters['em_iters']
batch_size = hyperparameters['batch_size']
learning_rate = hyperparameters['learning_rate']
train_split = hyperparameters['train_split']
val_split = hyperparameters['val_split']
test_split = hyperparameters['test_split']
hidden_units = hyperparameters['hidden_units']

print("Reading in data...")
train_loader, val_loader, test_loader = create_data_loaders(data_path, batch_size)

total_steps = epochs * len(train_loader)
data = []
for batch in train_loader:
    data.append(batch)
data = torch.cat(data, dim=0)


def sample_base(base, N, with_noise):
    if type(base) == LowRankMixtureModel:
        samples = base.sample(N, with_noise=with_noise)[0]
    else:
        samples = base.sample((N,))
        
    return samples

# %%
if flow == 'fm':
    flow_matcher = ConditionalFlowMatcher(sigma=sigma)
elif flow == 'ot-fm':
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
else:
    raise ValueError

model = MLP(dim=n_features, w=hidden_units, time_varying=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

early_stopping = EarlyStopping(patience=10, delta=1e-4, verbose=True)

start = time.time()

if base == 'normal':
    base_distribution = MultivariateNormal(
        torch.zeros(n_features).to(device), 
        torch.eye(n_features).to(device)
    )
elif base == 'mppca':
    base_distribution = LowRankMixtureModel(
        n_components=n_components,
        n_features=n_features,
        n_factors=n_factors,
        init_method='kmeans'
    ).to(device)
    mppca_lp = base_distribution.fit(data, max_iterations=em_iters, feature_sampling=False)
else:
    raise ValueError

#%%
def compute_loss(x0, x1, flow_matcher, model):
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
    vt = model(torch.cat([xt, t[:, None]], dim=-1))
    loss = torch.mean((vt - ut) ** 2)

    return loss


for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        x0 = sample_base(base=base_distribution, N=batch_size, with_noise=True).to(device)
        x1 = batch.to(device)
        loss = compute_loss(x0, x1, flow_matcher, model)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x0 = sample_base(base=base_distribution, N=batch_size, with_noise=True).to(device)
            x1 = batch.to(device)
            val_loss += compute_loss(x0, x1, flow_matcher, model).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Val Log-Likelihood: {val_loss:.4f}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Stopping early at epoch {epoch + 1}")
        break

end = time.time()

print("Total training time: {:0.2f} s".format(end - start))

model.eval()
avg_lps = []
nfes = []
for batch in tqdm(test_loader):
    avg_lp, nfe = cnf_test_metrics(model, batch, device, base_distribution, autograd_trace)
    avg_lps.append(avg_lp)
    nfes.append(nfe)
# %%
