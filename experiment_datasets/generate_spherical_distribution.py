import numpy as np
from typing import List, Tuple
from experiment_datasets import Dataset


def generate_spherical_distribution(
    n: int, true_theta: np.ndarray, delta: float = 0.2  # Same value as in article
) -> Dataset:
    """
    Generate spherical distribution data
    """
    mu = true_theta[:-1].copy()
    r = true_theta[-1]

    # U is randomly generated on the unit sphere
    U = np.random.randn(n, mu.shape[0])
    U /= np.linalg.norm(U, axis=1)[:, None]

    W = np.random.uniform(1 - delta, 1 + delta, n)
    X = mu + r * U * W[:, None]
    return Dataset(X=X)
