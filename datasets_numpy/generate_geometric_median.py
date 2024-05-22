import numpy as np
from typing import List, Tuple
from datasets_numpy import MyDataset


def generate_covariance_matrix(d, eigenvalues) -> np.ndarray:
    """Generate a covariance matrix with specified eigenvalues."""
    # Create a diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)

    # Generate a random orthogonal matrix Q
    Q, R = np.linalg.qr(np.random.standard_normal((d, d)))

    # Ensure Q is truly orthogonal
    Q = Q @ np.diag(np.sign(R.diagonal()))

    # Generate the covariance matrix
    C = Q @ D @ Q.T
    return C


def generate_geometric_median(
    n: int, true_theta: np.ndarray, eigenvalues: List[float] = None
) -> Tuple[MyDataset, str]:
    """
    Generate data from a multivariate normal distribution with specified eigenvalues.
    """
    name = "multivariate normal"

    d = len(true_theta)
    # if eigenvalues is None:
    #     eigenvalues = np.logspace(-2, 2, d)
    # covariance_matrix = generate_covariance_matrix(d, eigenvalues)
    covariance_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            covariance_matrix[i, j] = 0.5 ** abs(i - j)

    X = np.random.multivariate_normal(mean=true_theta, cov=covariance_matrix, size=n)
    return MyDataset(X=X), name
