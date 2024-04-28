import numpy as np
from typing import Tuple

from objective_functions import BaseObjectiveFunction


def sigmoid(x: np.ndarray):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseObjectiveFunction):
    def __call__(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        X = np.atleast_2d(X)
        n = X.shape[0]
        phi = np.hstack([np.ones(n, 1), X])
        dot_product = np.dot(phi, h)
        return np.log(1 + np.exp(dot_product)) - dot_product * Y

    def grad(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X])
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        # grad = (p - Y)[:, np.newaxis] * phi
        grad = (p - Y) * phi  # Equivalent
        return grad

    def grad_and_hessian(
        self, X: np.ndarray, Y: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
