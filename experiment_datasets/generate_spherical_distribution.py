import torch
from torch.utils.data import TensorDataset


def generate_spherical_distribution(
    n: int, true_theta: torch.Tensor, delta: float = 0.2  # Same value as in article
) -> TensorDataset:
    """
    Generate spherical distribution data
    """
    mu = true_theta[:-1].copy()
    r = true_theta[-1]

    # U is randomly generated on the unit sphere
    U = torch.randn(n, mu.size(0))
    U /= U.norm(dim=1, keepdim=True)

    # W is a random scaling factor to vary the radius slightly
    W = torch.empty(n).uniform_(1 - delta, 1 + delta)

    X = mu + r * U * W.unsqueeze(1)

    return TensorDataset(X)
