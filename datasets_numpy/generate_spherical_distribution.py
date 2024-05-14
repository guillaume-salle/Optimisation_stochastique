import numpy as np
from typing import Tuple
from datasets_numpy import MyDataset


def generate_spherical_distribution(
    n: int, true_theta: np.ndarray, delta: float = 0.2  # Same value as in article
) -> Tuple[MyDataset, str]:
    """
    Generate spherical distribution data
    """
    name = "spherical distribution"

    mu = true_theta[:-1]
    r = true_theta[-1]

    # U is randomly generated on the unit sphere
    U = np.random.randn(n, mu.shape[0])
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # W is a random scaling factor to vary the radius slightly
    W = np.random.uniform(1 - delta, 1 + delta, n)

    X = mu + r * U * W[:, None]

    return MyDataset(X=X), name
