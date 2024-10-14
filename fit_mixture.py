import argparse
import sys, os
import torch
from torchvision.datasets import CelebA, MNIST, CIFAR10
import torchvision.transforms as transforms
import numpy as np
from models import LowRankMixtureModel
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_model
from matplotlib import pyplot as plt
import time

from PIL import Image

"""
MFA model training (data fitting) example.
Note that actual EM (and SGD) training code are part of the MFA class itself.
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'celeba', 'cifar10'])
    parser.add_argument('--fit_method', type=str, default='batch_em',
                        choices=['batch_em', 'em'])
    args = parser.parse_args()


    #dataset = argv[1] if len(argv) == 2 else 'celeba'
    #dataset = 'cifar10'
    print('Preparing dataset and parameters for', args.dataset, '...')

    if args.dataset == 'celeba':
        image_shape = [64, 64, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 10              # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'rnd_samples'     # Initialize each component from few random samples using PPCA
        trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]),
                                    transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = CelebA(root='./data', split='train', transform=trans, download=True)
        test_set = CelebA(root='./data', split='test', transform=trans, download=True)
    elif args.dataset == 'cifar10':
        image_shape = [32, 32, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 2000               # The EM batch size
        num_iterations = 20             # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'rnd_samples'     # Initialize each component from few random samples using PPCA
        trans = transforms.Compose(
            [
                transforms.Resize(image_shape[0]),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ReshapeTransform([-1])
            ]
        )
        train_set = CIFAR10(root='./data', train=True, transform=trans, download=True)
        test_set = CIFAR10(root='./data', train=False, transform=trans, download=True)
    elif args.dataset == 'mnist':
        image_shape = [28, 28]          # The input image shape
        n_components = 50               # Number of components in the mixture model
        n_factors = 6                   # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 20             # Number of EM iterations (=epochs)
        feature_sampling = False       # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'kmeans'         # Initialize by using k-means clustering
        trans = transforms.Compose([transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = MNIST(root='./data', train=True, transform=trans, download=True)
        test_set = MNIST(root='./data', train=False, transform=trans, download=True)
    else:
        assert False, 'Unknown dataset: ' + args.dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = 'cpu'

    model_dir = './models/'+args.dataset
    os.makedirs(model_dir, exist_ok=True)
    figures_dir = './figures/'+args.dataset
    os.makedirs(figures_dir, exist_ok=True)
    model_name = 'c_{}_l_{}_init_{}'.format(n_components, n_factors, init_method)

    print('Defining the MFA model...')
    model = LowRankMixtureModel(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors,
                init_method=init_method).to(device=device)
    
    print('EM fitting: {} components / {} factors / batch size {} ...'.format(n_components, n_factors, batch_size))

    if args.fit_method == "batch_em":
        start = time.time()
        ll_log = model.batch_fit(train_set, test_set, batch_size=batch_size, max_iterations=num_iterations,
                                feature_sampling=feature_sampling)
        end = time.time()
        print("time {:0.2f}".format(end-start))  
    elif args.fit_method == "em":
        images = []
        labels = []

        for img, label in train_set:
            images.append(img)
            labels.append(label)

        # Stack images and labels into tensor format
        images_tensor = torch.stack(images)

        start = time.time()
        ll_log = model.fit(images_tensor, max_iterations=num_iterations, feature_sampling=feature_sampling)
        end = time.time()
        print("time {:0.2f}".format(end-start))
    else:
        assert False, 'Unknown fit method: ' + args.fit_method

    if mfa_sgd_epochs > 0:
        print('Continuing training using SGD with diagonal (instead of isotropic) noise covariance...')
        model.isotropic_noise = False
        ll_log_sgd = model.sgd_mfa_train(train_set, test_set, batch_size=1000, test_size=256, max_epochs=mfa_sgd_epochs,
                                         feature_sampling=feature_sampling)
        ll_log += ll_log_sgd

    print('Saving the model...')
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_'+model_name+'.pth'))

    #model.to('cpu')
    print('Visualizing the trained model...')
    model_image = visualize_model(model, image_shape=image_shape, end_component=10)
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

if __name__ == "__main__":
    main()