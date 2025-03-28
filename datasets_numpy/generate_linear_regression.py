import numpy as np
from typing import Tuple
from datasets_numpy import MyDataset


def toeplitz_matrix(n: int, const: float = 0.9, diag: bool = False) -> np.ndarray:
    """
    Generate a Toeplitz matrix of size n with a modified diagonal

    Parameters:
    n (int): Size of the matrix.
    const (float): Value to be taken to the power of the absolute difference between the indices.
    diag (bool): Whether the diagonal should be modified.

    Returns:
    np.ndarray: Toeplitz matrix.
    """
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = const ** abs(i - j)
            else:
                matrix[i, j] = (1 + i) if diag else 1
    if diag:
        matrix /= n
    return matrix


def generate_linear_regression(
    n: int,
    true_theta: np.ndarray,
    bias: bool = True,
    toeplitz: bool = False,
    const_toeplitz: float = 0.9,
    diag: bool = False,
) -> Tuple[MyDataset, str]:
    """
    Generate data from a linear regression model.
    """
    name = (
        "linear regression"
        + (f" Toeplitz {const_toeplitz}" if toeplitz else "")
        + (" diag" if diag else "")
    )

    d = len(true_theta) - 1 if bias else len(true_theta)
    if toeplitz:
        covariance_matrix = toeplitz_matrix(d, const=const_toeplitz, diag=diag)
    else:
        covariance_matrix = np.eye(d)

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=covariance_matrix, size=n)
    if bias:
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        phi = X

    Y = phi @ true_theta + np.random.standard_normal((n))

    return MyDataset(X=X, Y=Y), name
