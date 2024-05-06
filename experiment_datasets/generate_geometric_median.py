import numpy as np
from typing import List, Tuple
from experiment_datasets import Dataset


def generate_geometric_median(
    n: int, true_theta: np.ndarray, cov: str = "exponential"
) -> Dataset:
    """
    Generate data from a multivariate normal distribution with a given mean and covariance matrix.
    """
    if cov not in ["exponential", "article"]:
        raise ValueError("Invalid cov argument, must be 'exponential' or 'article'")
    d = len(true_theta)
    covariance_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if cov == "exponential":
                covariance_matrix[i, j] = np.exp(-abs(i - j))
            elif cov == "article":
                covariance_matrix[i, j] = abs(i - j) ** 0.5

    X = np.random.multivariate_normal(mean=true_theta, cov=covariance_matrix, size=n)
    return Dataset(X=X)
