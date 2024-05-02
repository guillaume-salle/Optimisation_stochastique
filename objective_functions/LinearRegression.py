import numpy as np
from typing import List, Tuple

from objective_functions import BaseObjectiveFunction


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear model"

    def __call__(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X = np.atleast_2d(X)
        phi = np.hstack([np.ones((X.shape[0], 1)), X]) if self.bias else X
        Y_pred = np.dot(phi, theta)
        error = Y_pred - Y
        return 0.5 * np.dot(error, error)

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1] + 1 if self.bias else X.shape[-1]

    def grad(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the linear regression loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        Y_pred = np.dot(phi, theta)
        error = Y_pred - Y
        return error * phi

    def hessian(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        return np.outer(phi, phi)

    def grad_and_hessian(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        Y_pred = np.dot(phi, theta)
        error = Y_pred - Y
        grad = error * phi
        hessian = np.outer(phi, phi)
        return grad, hessian

    def grad_and_riccati(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Riccati of the linear regression loss, works only for a single data point
        """
        phi = np.hstack([np.ones((1,)), X]) if self.bias else X
        Y_pred = np.dot(phi, theta)
        error = Y_pred - Y
        grad = error * phi
        riccati = phi
        return grad, riccati
