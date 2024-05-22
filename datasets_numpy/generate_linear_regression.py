import numpy as np
from typing import Tuple
from datasets_numpy import MyDataset


def generate_linear_regression(
    n: int, true_theta: np.ndarray, bias: bool = True
) -> Tuple[MyDataset, str]:
    """
    Generate data from a linear regression model.
    """
    name = "linear regression"

    d = len(true_theta)
    if bias:
        X = np.random.standard_normal((n, d - 1))
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        X = np.random.standard_normal((n, d))
        phi = X

    Y = phi @ true_theta + np.random.standard_normal((n))

    return MyDataset(X=X, Y=Y), name
