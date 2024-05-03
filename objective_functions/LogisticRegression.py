import numpy as np
from typing import Tuple

from objective_functions import BaseObjectiveFunction
from datasets import Dataset


def sigmoid(z: float):
    """Stable sigmoid function that avoids overflow."""
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def sigmoid_array(z):
    """Compute the sigmoid function in a stable way for arrays."""
    # positive_mask = np.zeros_like(z, dtype=bool)
    # positive_mask[z >= 0] = True
    positive_mask = z >= 0
    negative_mask = ~positive_mask

    sigmoid = np.zeros_like(z, dtype=float)

    # Positive elements
    exp_neg = np.exp(-z[positive_mask])
    sigmoid[positive_mask] = 1 / (1 + exp_neg)

    # Negative elements
    exp_pos = np.exp(z[negative_mask])
    sigmoid[negative_mask] = exp_pos / (1 + exp_pos)

    return sigmoid


def sigmoid_array(z):
    """Compute the sigmoid function in a stable way for arrays."""
    positive_mask = z >= 0
    negative_mask = ~positive_mask

    sigmoid = np.zeros_like(z, dtype=float)  # Ensuring the array is of float type

    # Positive elements
    exp_neg = np.exp(-z[positive_mask])
    sigmoid[positive_mask] = 1 / (1 + exp_neg)

    # Negative elements
    exp_pos = np.exp(z[negative_mask])
    sigmoid[negative_mask] = exp_pos / (1 + exp_pos)

    return sigmoid


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class
    """

    def __init__(self, bias: bool = True):
        self.name = "Logistic model"
        self.bias = bias

    def __call__(self, X: Tuple, h: np.ndarray) -> np.ndarray:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        x, y = X
        x = np.atleast_2d(x)
        phi = np.hstack([np.ones((x.shape[0], 1)), x]) if self.bias else x
        dot_product = np.dot(phi, h)
        return np.log(1 + np.exp(dot_product)) - dot_product * y

    def evaluate_accuracy(self, dataset: Dataset, h: np.ndarray) -> float:
        """
        Compute the accuracy of the model
        """
        X, Y = dataset.X, dataset.Y
        X = np.atleast_2d(X)
        phi = np.hstack([np.ones((X.shape[0], 1)), X]) if self.bias else X
        dot_product = np.dot(phi, h)
        p = sigmoid_array(dot_product)
        return round(100 * np.mean((p > 0.5) == Y), 2)

    def get_theta_dim(self, X: Tuple) -> int:
        """
        Return the dimension of theta
        """
        x, _ = X
        return x.shape[-1] + 1 if self.bias else x.shape[-1]

    def grad(self, X: Tuple, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic loss, works only for a single data point
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        # grad = (p - Y)[:, np.newaxis] * phi
        grad = (p - y) * phi  # Equivalent
        return grad

    def hessian(self, X: Tuple, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the logistic loss, works only for a single data point
        """
        x, _ = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        hessian = p * (1 - p) * np.outer(phi, phi)
        return hessian

    def grad_and_hessian(
        self, X: Tuple, h: np.ndarray
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
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        grad = (p - y) * phi
        hessian = p * (1 - p) * np.outer(phi, phi)
        return grad, hessian

    def grad_and_riccati(
        self, X: Tuple, h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Ricatti of the logistic loss
        Does not work for a batch of data because of the outer product
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        dot_product = np.dot(phi, h)
        p = sigmoid(dot_product)
        grad = (p - y) * phi
        ricatti = np.sqrt(p * (1 - p)) * phi
        return grad, ricatti
