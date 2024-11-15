import argparse
import os
import torch
from torchvision.datasets import CelebA, MNIST, CIFAR10, FGVCAircraft
import torchvision.transforms as transforms
import numpy as np
from models import LowRankMixtureModel
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_mixture
from matplotlib import pyplot as plt
import time

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'celeba', 'cifar10', 'fgvc-aircraft'])
parser.add_argument('--fit_method', type=str, default='batch_em',
                    choices=['batch_em', 'em'])
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--n_factors', type=int, default=5)
args = parser.parse_args()

print('Preparing dataset and parameters for', args.dataset,'...')

if args.dataset == 'celeba':
    image_shape = [64, 64, 3]
    n_components = args.n_components
    n_factors = args.n_factors
    batch_size = 1000
    num_iterations = 5
    feature_sampling = 0.3
    init_method = 'rnd_samples'
    trans = transforms.Compose(
        [
            CropTransform((25, 50, 25+128, 50+128)), 
            transforms.Resize(image_shape[0]),
            transforms.ToTensor(),  
            ReshapeTransform([-1])
        ]
    )
    train_set = CelebA(root='./data', split='train', transform=trans, download=True)
    test_set = CelebA(root='./data', split='test', transform=trans, download=True)

elif args.dataset == 'cifar10':
    image_shape = [32, 32, 3]
    n_components = args.n_components
    n_factors = args.n_factors
    batch_size = 2000
    num_iterations = 10
    feature_sampling = 0.3
    init_method = 'rnd_samples'
    trans = transforms.Compose(
        [
            transforms.Resize(image_shape[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ReshapeTransform([-1])
        ]
    )
    train_set = CIFAR10(root='./data', train=True, transform=trans, download=True)
    test_set = CIFAR10(root='./data', train=False, transform=trans, download=True)

elif args.dataset == "fgvc-aircraft":
    image_shape = [64, 64, 3]
    n_components = args.n_components
    n_factors = args.n_factors
    batch_size = 200
    num_iterations = 3
    feature_sampling = 0.3
    init_method = 'rnd_samples'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.Resize(image_shape[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ReshapeTransform([-1])
        ]
    )
    train_set = FGVCAircraft(root='./data', split = 'trainval', transform=trans, download=True)
    test_set = FGVCAircraft(root='./data', split='test', transform=trans, download=True)

elif args.dataset == 'mnist':
    image_shape = [28, 28]
    n_components = args.n_components
    n_factors = args.n_factors
    batch_size = 1000
    num_iterations = 20
    feature_sampling = False
    init_method = 'kmeans'
    trans = transforms.Compose(
        [
            transforms.ToTensor(),  
            ReshapeTransform([-1])
        ]
    )
    train_set = MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = MNIST(root='./data', train=False, transform=trans, download=True)

else:
    assert False, 'Unknown dataset: ' + args.dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_dir = './models/' + args.dataset
os.makedirs(model_dir, exist_ok=True)
figures_dir = './figures/' + args.dataset
os.makedirs(figures_dir, exist_ok=True)
model_name = 'c_{}_l_{}'.format(n_components, n_factors)

print('Defining the MFA model...')
model = LowRankMixtureModel(
            n_components=n_components, 
            n_features=np.prod(image_shape), 
            n_factors=n_factors,
            init_method=init_method
        ).to(device=device)

print('EM fitting: {} components / {} factors / batch size {}...'.format(
    n_components, n_factors, batch_size))

if args.fit_method == "batch_em":
    start = time.time()
    ll_log = model.batch_fit(
                        train_dataset=train_set, 
                        test_dataset=test_set, 
                        batch_size=batch_size, 
                        max_iterations=num_iterations,
                        feature_sampling=feature_sampling)
    end = time.time()
    print("time {:0.2f}".format(end-start)) 

elif args.fit_method == "em":
    images = []
    labels = []

    for img, label in train_set:
        images.append(img)
        labels.append(label)

    # stack images and labels into tensor format
    images_tensor = torch.stack(images)

    start = time.time()
    ll_log = model.fit(
                    x=images_tensor, 
                    max_iterations=num_iterations, 
                    feature_sampling=feature_sampling)
    end = time.time()
    print("time {:0.2f}".format(end-start))
else:
    assert False, 'Unknown fit method: ' + args.fit_method

print('Saving the model...')
torch.save(model.state_dict(), os.path.join(model_dir, 'model_'+model_name+'.pth'))

model.to('cpu')
print('Visualizing the trained model...')
model_image = visualize_mixture(model, image_shape=image_shape, end_component=5)
image = Image.fromarray((255 * model_image).astype(np.uint8))
image.save(os.path.join(figures_dir, 'model_'+model_name+'.png'))

print('Generating random samples...')
rnd_samples, _ = model.sample(100, with_noise=False)
mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
image = Image.fromarray((255 * mosaic).astype(np.uint8))
image.save(os.path.join(figures_dir, 'samples_'+model_name+'.png'))

print('Plotting test log-likelihood graph...')
plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'training_graph_'+model_name+'.png'))
print('Done.')