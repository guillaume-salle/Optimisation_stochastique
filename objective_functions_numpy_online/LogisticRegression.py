import numpy as np
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction, add_bias
from datasets_numpy import MyDataset


def sigmoid(z: float):
    """Stable sigmoid function that avoids overflow."""
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def sigmoid_array(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function in a stable way for arrays."""
    sigmoid = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    return sigmoid


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class
    """

    def __init__(self, bias: bool = True):
        self.name = "Logistic model"
        self.bias = bias

    def __call__(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> np.ndarray:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        X, y = data
        X = np.atleast_2d(X)
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        return np.log(1 + np.exp(dot_product)) - dot_product * y

    def evaluate_accuracy(
        self, dataset: MyDataset, h: np.ndarray, batch_size=512
    ) -> float:
        """
        Compute the accuracy of the model
        """
        X, Y = dataset.X, dataset.Y
        correct_predictions = 0
        total = 0

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            if self.bias:
                X_batch = add_bias(X_batch)
            dot_product = np.dot(X_batch, h)
            p = sigmoid_array(dot_product)
            predictions = (p > 0.5).astype(int)
            correct_predictions += np.sum(predictions == Y_batch)
            total += Y_batch.shape[0]

        return correct_predictions / total

    def get_theta_dim(self, data: Tuple[np.ndarray, np.ndarray]) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        if self.bias:
            return X.shape[-1] + 1
        else:
            return X.shape[-1]

    def grad(self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic loss, works only for a single data point
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        grad = (p - y) * X
        return grad

    def hessian(self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the logistic loss, works only for a single data point
        """
        X, _ = data
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        hessian = p * (1 - p) * np.outer(X, X)
        return hessian

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the logistic loss
        Does not work for a batch of data because of the outer product
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        grad = (p - y) * X
        hessian = p * (1 - p) * np.outer(X, X)
        return grad, hessian

    def grad_and_riccati(
        self, data: Tuple, h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Ricatti of the logistic loss
        Does not work for a batch of data because of the outer product
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        grad = (p - y) * X
        ricatti = np.sqrt(p * (1 - p)) * X
        return grad, ricatti
