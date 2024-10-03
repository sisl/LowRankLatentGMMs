# Toy datasets adapted from the source at "Invertible Residual Networks":
# https://github.com/rtqichen/residual-flows

#*******************************************************************************
# IMPORTS AND SETUPS
#*******************************************************************************
import numpy as np
import sklearn.datasets as datasets

#*******************************************************************************
# FUNCTION DEFINITIONS
#*******************************************************************************
def generate_data(target, rng=None, batch_size=256):
    if rng is None:
        rng = np.random.RandomState()

    if target == "moons":
        X = datasets.make_moons(n_samples=batch_size, noise=0.08)[0]
        X = X.astype("float32")
        X = X * 2 + np.array([-1, -0.2])
        return X

    elif target == "swissroll":
        X = datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        X = X.astype("float32")[:, [0, 2]]
        X /= 5
        return X

    elif target == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0), 
            (-1, 0), 
            (0, 1), 
            (0, -1), 
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)), 
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        X = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            X.append(point)
        X = np.array(X, dtype="float32")
        X /= 1.414
        return X

    elif target == "checkerboard":
        X1 = np.random.rand(batch_size) * 4 - 2
        X2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        X2 = X2_ + (np.floor(X1) % 2)
        return np.concatenate([X1[:, None], X2[:, None]], 1) * 2

    elif target == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
    
    else:
        return generate_data("moons", batch_size)