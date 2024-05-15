import numpy as np


def add_bias_1d(X: np.ndarray) -> np.ndarray:
    """
    Add a bias term (1) at the beginning of a 1D array.
    """
    return np.insert(X, 0, 1)


def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add a bias term to the input data
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])
