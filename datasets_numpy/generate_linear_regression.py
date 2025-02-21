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

    CONST_TOEPLITZ = 0.9

    d = len(true_theta) - 1 if bias else len(true_theta)
    covariance_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                covariance_matrix[i, j] = CONST_TOEPLITZ ** abs(i - j)
            else:
                covariance_matrix[i, j] = 1 + i
                # covariance_matrix[i, j] = 1

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=covariance_matrix, size=n)
    if bias:
        phi = np.hstack([np.ones((n, 1)), X])
    else:
        phi = X

    Y = phi @ true_theta + np.random.standard_normal((n))

    return MyDataset(X=X, Y=Y), name
