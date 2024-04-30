import numpy as np
from typing import List, Tuple


def generate_linear_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    d = len(true_theta)
    if bias:
        X = np.random.randn(n, d - 1)
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.randn(n, d)
        phi = X
    Y = phi @ true_theta + np.random.randn(n)
    return list(zip(X, Y))
