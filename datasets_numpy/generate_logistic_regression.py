import numpy as np
import torch
from typing import List, Tuple
from datasets_numpy import MyDataset


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
    n: int, true_theta: np.ndarray, bias: bool = True
) -> Tuple[MyDataset, str]:
    """
    Generate data from a linear regression model.
    """
    name = "logistic regression"

    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X

    Y = np.random.binomial(1, sigmoid_torch(phi @ true_theta))

    return MyDataset(X=X, Y=Y), name
