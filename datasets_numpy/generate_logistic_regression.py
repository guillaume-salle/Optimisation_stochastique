import numpy as np
from typing import List, Tuple
from datasets_numpy import MyDataset


def sigmoid_array(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function in a stable way for arrays."""
    sigmoid = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return sigmoid


def generate_logistic_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> Tuple[MyDataset, str]:
    """
    Generate data from a linear regression model.
    """
    name = "linear regression"

    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X

    Y = np.random.binomial(1, sigmoid_array(phi @ true_theta))

    return MyDataset(X=X, Y=Y), name
