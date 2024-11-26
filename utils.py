import glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os
from PIL import Image
import torch

from torchcfm.utils import torch_wrapper
from torchdyn.core import DEFunc, NeuralODE
from cnf import compute_log_probs

from models import LowRankMixtureModel

# color-blind friendly palette
pastelBlue = "#0072B2"
pastelRed = "#F5615C"


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def samples_to_np_images(samples, image_shape=[64, 64, 3], clamp=True):
    assert len(samples.shape) == 2
    assert samples.shape[1] == np.prod(image_shape)
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)
    samples_out = samples if not clamp else torch.clamp(samples, 0., 1.)
    if len(image_shape) == 3:
        return samples_out.reshape(-1, image_shape[2], image_shape[0], image_shape[1]).permute(0, 2, 3, 1).cpu().numpy()
    else:
        return samples_out.reshape(-1, image_shape[0], image_shape[1]).cpu().numpy()


def sample_to_np_image(sample, image_shape=[64, 64, 3]):
    return samples_to_np_images(sample.unsqueeze(0), image_shape).squeeze()


def samples_to_mosaic(samples, image_shape=[64, 64, 3]):
    images = samples_to_np_images(samples, image_shape)
    num_images = images.shape[0]
    num_cols = int(np.ceil(np.sqrt(num_images)))
    rows = []
    for i in range(num_images // num_cols):
        rows.append(np.hstack([images[j] for j in range(i*num_cols, (i+1)*num_cols)]))
    return np.vstack(rows)


def visualize_mixture(model, image_shape=[64, 64, 3], start_component=0, end_component=None):
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)
    K, d, l = model.W.shape
    h, w = image_shape[:2]
    spacer = min(8, w//8)
    end_component = end_component or min(K, 2048//(w*3+2+spacer))
    k = end_component - start_component
    z = 1.5

    def to_im(x):
        return sample_to_np_image(x, image_shape=image_shape)

    if len(image_shape) == 3:
        canvas = np.ones([(l+1)*(h+1), k*(w*3+2) + (k-1)*spacer, image_shape[2]])
    else:
        canvas = np.ones([(l+1)*(h+1), k*(w*3+2) + (k-1)*spacer])
    for c_num in range(start_component, end_component):
        x_start = (c_num-start_component)*(w*3+2+spacer)

        mu = model.mu[c_num]
        canvas[:h, x_start+w//2:x_start+w//2+w] = to_im(mu)

        D = torch.exp(0.5*model.log_Psi[c_num])
        canvas[:h, x_start+w//2+w+2:x_start+w//2+2*w+2] = to_im(D / torch.max(D))

        for i in range(l):
            y_start = (i+1)*(h+1)
            A_i = model.W[c_num, :, i]
            canvas[y_start:y_start+h, x_start:x_start+w] = to_im(mu + z * A_i)
            canvas[y_start:y_start+h, x_start+w+1:x_start+2*w+1] = to_im(0.5 + z * A_i)
            canvas[y_start:y_start+h, x_start+2*w+2:x_start+3*w+2] = to_im(mu - z * A_i)
    return canvas

def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=1, c='k')
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=pastelRed)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=10, alpha=1, c=pastelRed)
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.gca().set_aspect('equal')
    plt.show()

def visualize_model(model, base, title):
    w = 4
    points = 200j
    device = "cpu"
    Y, X = np.mgrid[-w:w:points, -w:w:points]
    gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32)
    points_small = 20j
    points_real_small = 20
    Y_small, X_small = np.mgrid[-w:w:points_small, -w:w:points_small]
    gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
        torch.float32
    )

    torch.manual_seed(42)
    sample = base.sample((1024,))
    ts = torch.linspace(0, 1, 51)

    nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler").to(device)
    # with torch.no_grad():
    traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()

    for i, t in enumerate(ts):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # density plot
        with torch.no_grad():
            if t > 0:
                log_probs = compute_log_probs(model, gridpoints, t, device, base)
            else:
                log_probs = base.log_prob(gridpoints)

        log_probs = log_probs.reshape(Y.shape)
        ax = axes[0]
        ax.pcolormesh(X, Y, torch.exp(log_probs))

        # Quiver plot
        out = model(
            torch.cat(
                [gridpoints_small, torch.ones((gridpoints_small.shape[0], 1)) * t], dim=1
            ).to(device)
        )
        out = out.reshape([points_real_small, points_real_small, 2]).cpu().detach().numpy()
        ax = axes[1]
        ax.quiver(X_small, Y_small,
            out[:, :, 0],
            out[:, :, 1],
            np.sqrt(np.sum(out**2, axis=-1)),
            cmap="coolwarm",
            scale=15.0,
            width=0.01,
            pivot="mid",
        )

        # trajectories
        ax = axes[2]
        sample_traj = traj
        ax.scatter(sample_traj[0, :, 0], sample_traj[0, :, 1], s=15, alpha=1, c='k')
        ax.scatter(sample_traj[:i, :, 0], sample_traj[:i, :, 1], s=1, alpha=0.2, c=pastelRed)
        ax.scatter(sample_traj[i, :, 0], sample_traj[i, :, 1], s=15, alpha=1, c=pastelRed)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-w, w)
            ax.set_ylim(-w, w)
            ax.set_aspect('equal')
        plt.tight_layout()

        os.makedirs("figures/trajectory/{}/".format(title), exist_ok=True)
        plt.savefig("figures/trajectory/{}/{:0.2f}.png".format(title, t), dpi=100)
        plt.close()

def plot_pdf(distribution, ax):
    x = torch.linspace(-2, 2, 500)
    probs = distribution.log_prob(x).exp()
    norm = Normalize(vmin=probs.min(), vmax=probs.max())
    ax.scatter(x, probs, c=probs, s=2, norm=norm, cmap="inferno")
    ax.set_facecolor('black')
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 0.6)
    ax.set_aspect(2.0)

# only flip for FFJORD!
def plot_1d_trajectories(trajectories, base, lps, tspan, ax):
    probs = lps.exp()
    vmax = base.log_prob(torch.tensor(0)).exp()
    norm = Normalize(vmin=probs.min(), vmax=vmax)

    for i in range(trajectories.shape[1]):
        x = trajectories[:,i].squeeze().flip(-1)
        ax.scatter(x, tspan, s=1, c=probs[:,i], norm=norm, cmap="inferno")

    # Set the limits for the axes
    ax.set_facecolor('black')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(0.5)


def make_gif(frame_folder, out_path, delete_frames=True):
    files = [f for f in glob.glob(f"{frame_folder}/*.png")]
    #print(files)
    files = sorted(files)
    frames = [Image.open(image) for image in files]
    
    frame_one = frames[0]
    frame_one.save(out_path, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

    if delete_frames:
        for f in files:
            os.remove(f)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x
  

def sample_base(base, N, image_shape, with_noise):
    if type(base) == LowRankMixtureModel:
        samples = base.sample(N, with_noise=with_noise)[0].view(
            N, 3, image_shape[0], image_shape[1])
    else:
        samples = base.sample((N,)).view(
            N, image_shape[-1], image_shape[0], image_shape[1])
        
    return samples