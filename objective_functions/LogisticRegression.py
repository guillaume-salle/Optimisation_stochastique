import numpy as np
from typing import Tuple

from objective_functions import BaseObjectiveFunction


def sigmoid(x: np.ndarray):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class
    """

    def __init__(self, bias: bool = True):
        self.name = "Logistic model"
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        X = np.atleast_2d(X)
        phi = np.hstack([np.ones((X.shape[0], 1)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        return np.log(1 + np.exp(dot_product)) - dot_product * Y

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1] + 1 if self.bias else X.shape[-1]

    def grad(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        # grad = (p - Y)[:, np.newaxis] * phi
        grad = (p - Y) * phi  # Equivalent
        return grad

    def hessian(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the logistic loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        hessian = p * (1 - p) * np.outer(phi, phi)
        return hessian

    def grad_and_hessian(
        self, X: np.ndarray, Y: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the logistic loss
        Does not work for a batch of data because of the outer product
        """
        # For batch data, should work
        # n, d = X.shape
        # phi = np.hstack([np.ones(n, 1), X]) if self.bias else X
        # dot_product = np.dot(phi, h)
        # p = sigmoid(dot_product)
        # grad = (p - Y) * phi
        # hessian = np.einsum('i,ij,ik->ijk', p * (1 - p), phi, phi)
        # return grad, hessian

        # For a single data point
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        grad = (p - Y) * phi
        hessian = p * (1 - p) * np.outer(phi, phi)
        return grad, hessian

    def grad_and_riccati(
        self, X: np.ndarray, Y: np.ndarray, h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Ricatti of the logistic loss
        Does not work for a batch of data because of the outer product
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        grad = (p - Y) * phi
        ricatti = np.sqrt(p * (1 - p)) * phi
        return grad, ricatti
