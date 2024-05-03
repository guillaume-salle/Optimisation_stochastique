import numpy as np
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear model"

    def __call__(self, X: Any, theta: np.ndarray) -> np.ndarray:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        x, y = X
        x = np.atleast_2d(x)
        phi = np.hstack([np.ones((x.shape[0], 1)), x]) if self.bias else x
        Y_pred = np.dot(phi, theta)
        error = Y_pred - y
        return 0.5 * np.dot(error, error)

    def get_theta_dim(self, X: Any) -> int:
        """
        Return the dimension of theta
        """
        x, _ = X
        return x.shape[-1] + 1 if self.bias else x.shape[-1]

    def grad(self, X: Any, theta: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the linear regression loss, works only for a single data point
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        Y_pred = np.dot(phi, theta)
        error = Y_pred - y
        return error * phi

    def hessian(self, X: Any, theta: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, works only for a single data point
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        return np.outer(phi, phi)

    def grad_and_hessian(
        self, X: Any, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss, works only for a single data point
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        Y_pred = np.dot(phi, theta)
        error = Y_pred - y
        grad = error * phi
        hessian = np.outer(phi, phi)
        return grad, hessian

    def grad_and_riccati(
        self, X: Any, theta: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Riccati of the linear regression loss, works only for a single data point
        """
        x, y = X
        phi = np.hstack([np.ones((1,)), x]) if self.bias else x
        Y_pred = np.dot(phi, theta)
        error = Y_pred - y
        grad = error * phi
        riccati = phi
        return grad, riccati
