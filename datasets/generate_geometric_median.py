import numpy as np
from typing import List, Tuple


def generate_geometric_median(
    n: int, true_theta: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    d = len(true_theta)
    covariance_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            covariance_matrix[i, j] = abs(i - j) ** 0.5

    X = np.random.multivariate_normal(mean=true_theta, cov=covariance_matrix, size=n)
    return list(zip(X, None))
