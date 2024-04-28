import numpy as np

from objective_functions.logistic_regression import sigmoid


def create_dataset_logistic(n: int, true_theta: np.ndarray):
    d = len(true_theta)
    X = np.random.randn(n, d - 1)
    phi = np.hstack([np.ones((n, 1)), X])
    Y = np.random.binomial(1, sigmoid(phi @ true_theta))
    return list(zip(X, Y))  # Maybe use a dataloader for large datasets


def g_logistic(X: np.ndarray, Y: np.ndarray, h: np.ndarray):
    """
    Compute the logistic loss, works with a batch or a single data point
    """
    X = np.atleast_2d(X)
    n = X.shape[0]
    phi = np.hstack([np.ones(n, 1), X])
    dot_product = np.dot(phi, h)
    return np.log(1 + np.exp(dot_product)) - dot_product * Y


def g_grad_logistic(X: np.ndarray, Y: np.ndarray, h: np.ndarray):
    """
    Compute the gradient of the logistic loss, works only for a single data point
    """
    phi = np.hstack([np.ones((1,)), X])
    dot_product = np.dot(phi, h)
    p = sigmoid(dot_product)
    # grad = (p - Y)[:, np.newaxis] * phi
    grad = (p - Y) * phi  # Equivalent
    return grad


def g_grad_and_hessian_logistic(X, Y, h):
    """
    Compute the gradient and the Hessian of the logistic loss
    Does not work for a batch of data because of the outer product
    """
    # For batch data, should work
    # n, d = X.shape
    # phi = np.hstack([np.ones(n, 1), X])
    # dot_product = np.dot(phi, h)
    # p = sigmoid(dot_product)
    # grad = (p - Y) * phi
    # hessian = np.einsum('i,ij,ik->ijk', p * (1 - p), phi, phi)
    # return grad, hessian

    # For a single data point
    phi = np.hstack([np.ones((1,)), X])
    dot_product = np.dot(phi, h)
    p = sigmoid(dot_product)
    grad = (p - Y) * phi
    hessian = p * (1 - p) * np.outer(phi, phi)
    return grad, hessian
