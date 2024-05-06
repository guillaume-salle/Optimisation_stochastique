import numpy as np
from typing import List, Tuple
from experiment_datasets import Dataset


def sigmoid_array(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function in a stable way for arrays."""
    positive_mask = np.zeros_like(z, dtype=bool)
    positive_mask[z >= 0] = True
    negative_mask = ~positive_mask

    sigmoid = np.zeros_like(z, dtype=float)

    # Positive elements
    exp_neg = np.exp(-z[positive_mask])
    sigmoid[positive_mask] = 1 / (1 + exp_neg)

    # Negative elements
    exp_pos = np.exp(z[negative_mask])
    sigmoid[negative_mask] = exp_pos / (1 + exp_pos)

    return sigmoid


def generate_logistic_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> Dataset:
    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X
    Y = np.random.binomial(1, sigmoid_array(phi @ true_theta))
    return Dataset(X=X, Y=Y)
