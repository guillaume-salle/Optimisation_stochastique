import numpy as np


def sigmoid(x: np.ndarray):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def generate_logistic_dataset(n: int, true_theta: np.ndarray):
    d = len(true_theta)
    X = np.random.randn(n, d - 1)
    phi = np.hstack([np.ones((n, 1)), X])
    Y = np.random.binomial(1, sigmoid(phi @ true_theta))
    return list(zip(X, Y))  # Maybe use a dataloader for large datasets
