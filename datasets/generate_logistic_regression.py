import numpy as np
from typing import List, Tuple


def sigmoid(x: np.ndarray):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def generate_logistic_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> List[Tuple[np.ndarray, int]]:
    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X
    Y = np.random.binomial(1, sigmoid(phi @ true_theta))
    return list(zip(X, Y))
