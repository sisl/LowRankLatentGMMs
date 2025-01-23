#%%
import sys
import os
import argparse
from ndb import *
import numpy as np
import struct
from matplotlib import pyplot as plt
# from utils import image_batch_provider

from utils.datasets import ImageDataset


data_handler = ImageDataset(dataset="celeba", root_dir="./data", batch_size=256, image_shape=[64, 64, 3])


_, _, test_loader = data_handler.get_dataloaders()


def visualize_bins(bin_centers, is_different):
    k = bin_centers.shape[0]
    n_cols = 10
    n_rows = (k+n_cols-1)//n_cols
    for i in range(k):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(bin_centers[i, :].reshape([28, 28]))
        if is_different[i]:
            plt.plot([0, 27], [0, 27], 'r', linewidth=2)
        plt.axis('off')


#%%
import torch
data = []
for batch in test_loader:
    data.append(batch[0])
data = torch.cat(data, dim=0)
print(data.shape)

#%%
batch1 = data[:9856, ...]
batch2 = data[9856:, ...]


batch1 = batch1.reshape((9856, -1))
batch2 = batch2.reshape((9856, -1))


#%%




mnist_ndb = NDB(training_data=batch1, number_of_bins=100, z_threshold=4, whitening=False,
                cache_folder='./results/mnist_toy_example_ndb_cache')


results = mnist_ndb.evaluate(batch2, 'Validation')

# Visualize the missing bins
#visualize_bins(mnist_ndb.bin_centers, results['Different-Bins'])
# missing_bins = results['Proportions']/mnist_ndb.bin_proportions < 0.5
# visualize_bins(mnist_ndb.bin_centers, missing_bins)
#plt.savefig('bins_with_Val0-8_results_{}.png'.format(num_bins))

plt.figure()
mnist_ndb.plot_results()


# %%
