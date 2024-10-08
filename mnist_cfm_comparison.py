
import torch
from models import LowRankMixtureModel
import os
import numpy as np

image_shape = [28, 28]          # The input image shape
n_components = 50               # Number of components in the mixture model
n_factors = 6                   # Number of factors - the latent dimension (same for all components)
    
print('Loading pre-trained MFA model...')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = './models/' + 'mnist'
model = LowRankMixtureModel(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors).to(device=device)
model.load_state_dict(torch.load(os.path.join(model_dir, 'model_c_{}_l_{}.pth'.format(n_components, n_factors))))