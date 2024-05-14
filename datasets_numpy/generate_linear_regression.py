import numpy as np
from typing import List, Tuple
from experiment_datasets import MyDataset


def generate_linear_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> MyDataset:
    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X
    Y = phi @ true_theta + np.random.randn(n)
    return MyDataset(X=X, Y=Y)
