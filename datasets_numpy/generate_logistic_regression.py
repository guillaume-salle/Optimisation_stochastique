import numpy as np
import torch
from typing import Tuple
from datasets_numpy import MyDataset, toeplitz_matrix


def sigmoid_torch(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function using PyTorch.

    Parameters:
    z (np.ndarray): Input array.

    Returns:
    np.ndarray: Sigmoid of input array.
    """
    return torch.sigmoid(torch.as_tensor(z)).numpy()


def generate_logistic_regression(
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
    name = "logistic regression" + (" Toeplitz" if toeplitz else "") + (" + diag" if diag else "")

    d = len(true_theta)
    dim_X = d - 1 if bias else d
    if toeplitz:
        covariance_matrix = toeplitz_matrix(dim_X, const=const_toeplitz, diag=diag)
        X = np.random.multivariate_normal(mean=np.zeros(dim_X), cov=covariance_matrix, size=n)
    else:
        X = np.random.standard_normal((n, dim_X))
    phi = np.hstack([np.ones((n, 1)), X]) if bias else X

    Y = np.random.binomial(1, sigmoid_torch(phi @ true_theta))

    return MyDataset(X=X, Y=Y), name
