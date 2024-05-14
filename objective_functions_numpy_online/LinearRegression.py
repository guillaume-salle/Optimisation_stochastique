import numpy as np
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction, add_bias


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear model"

    def __call__(
        self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X, y = data
        X = np.atleast_2d(X)
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, theta)
        return 0.5 * (Y_pred - y) ** 2

    def get_theta_dim(self, data: Tuple[np.ndarray, np.ndarray]) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        if self.bias:
            return X.shape[-1] + 1
        else:
            return X.shape[-1]

    def grad(
        self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient of the linear regression loss, works only for a single data point
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, theta)
        grad = (Y_pred - y) * X
        return grad

    def hessian(
        self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, works only for a single data point
        """
        X, _ = data
        if self.bias:
            X = add_bias(X)
        return np.outer(X, X)

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss, works only for a single data point
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, theta)
        grad = (Y_pred - y) * X
        hessian = np.outer(X, X)
        return grad, hessian

    def grad_and_riccati(
        self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Riccati of the linear regression loss, works only for a single data point
        """
        X, y = data
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, theta)
        grad = (Y_pred - y) * X
        riccati = X
        return grad, riccati
