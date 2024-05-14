import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add a bias term to the input data
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])
