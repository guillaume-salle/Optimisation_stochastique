import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset
from typing import List, Tuple


def generate_covariance_matrix(d, eigenvalues) -> torch.Tensor:
    """Generate a covariance matrix with specified eigenvalues."""
    # Create a diagonal matrix of eigenvalues
    D = torch.diag(torch.tensor(eigenvalues))

    # Generate a random orthogonal matrix Q
    Q, R = torch.qr(torch.randn(d, d))

    # Ensure Q is truly orthogonal
    Q = Q @ torch.diag(R.diag().sign())

    # Generate the covariance matrix
    C = Q @ D @ Q.T
    return C


def generate_geometric_median(
    n: int,
    true_theta: torch.Tensor,
    # eigenvalues: List[float] = None,
) -> Tuple[TensorDataset, str]:
    """
    Generate data from a multivariate normal distribution with specified eigenvalues.
    """
    name = "multivariate normal"

    d = len(true_theta)
    # if eigenvalues is None:
    #     eigenvalues = torch.logspace(-2, 2, d)
    # covariance_matrix = generate_covariance_matrix(d, eigenvalues)
    covariance_matrix = torch.empty(d, d)
    for i in range(d):
        for j in range(d):
            covariance_matrix[i][j] = 0.5 ** abs(i - j)

    dist = MultivariateNormal(true_theta, covariance_matrix=covariance_matrix)
    X = dist.sample((n,))

    return TensorDataset(X), name
