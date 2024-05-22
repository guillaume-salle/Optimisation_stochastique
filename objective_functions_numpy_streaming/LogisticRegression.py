import numpy as np
import math
import torch
from typing import Tuple

from objective_functions_numpy_streaming import (
    BaseObjectiveFunction,
    add_bias,
    add_bias_1d,
)
from datasets_numpy import MyDataset


def sigmoid(z: float):
    """Stable sigmoid function that avoids overflow, works with a single value"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


def sigmoid_array(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function for array using PyTorch.

    Parameters:
    z (np.ndarray): Input array.

    Returns:
    np.ndarray: Sigmoid of input array.
    """
    return torch.sigmoid(torch.as_tensor(z)).numpy()


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
        if self.bias:
            if X.ndim == 1:
                X = add_bias_1d(X)
            else:
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
        Compute the gradient of the logistic loss, average over the batch
        """
        X, y = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid_array(dot_product)
        grad = np.dot(X.T, p - y) / n
        return grad

    def hessian(self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the logistic loss, average over the batch
        """
        X, _ = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid_array(dot_product)
        hessian = np.einsum("n,ni,nj->ij", p * (1 - p) / n, X, X)
        return hessian

    def hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray, col: int
    ) -> np.ndarray:
        """
        Compute a single column of the hessian of the objective function
        """
        X, _ = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid_array(dot_product)
        hessian_col = np.dot(X.T, p * (1 - p) * X[:, col]) / n
        return hessian_col

    def grad_and_hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the hessian of the objective function
        """
        X, y = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid_array(dot_product)
        grad = np.dot(X.T, p - y) / n
        hessian_col = np.dot(X.T, p * (1 - p) * X[:, col]) / n
        return grad, hessian_col

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the logistic loss, average over the batch
        """
        X, y = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        dot_product = np.dot(X, h)
        p = sigmoid_array(dot_product)
        grad = np.dot(X.T, p - y) / n
        hessian = np.einsum("n,ni,nj->ij", p * (1 - p) / n, X, X)
        return grad, hessian

    def riccati(
        self, data: Tuple, h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the ricatti term of the logistic loss, works only with a batch of size 1
        """
        X, _ = data
        n = X.shape[0]
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        ricatti = math.sqrt(p * (1 - p)) * X
        # alpha = max(math.sqrt(p * (1 - p)), 1.0 / iter**0.25)  # cf article bercu
        # riccati = alpha * X
        return ricatti

    def grad_and_riccati(
        self, data: Tuple, h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Ricatti of the logistic loss,
        works only with a batch of size 1
        """
        X, y = data
        n = X.shape[0]
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        dot_product = np.dot(X, h)
        p = sigmoid(dot_product)
        grad = (p - y) * X
        ricatti = math.sqrt(p * (1 - p)) * X
        # alpha = max(math.sqrt(p * (1 - p)), 1.0 / iter**0.25)  # cf article bercu
        # riccati = alpha * X
        return grad, ricatti
