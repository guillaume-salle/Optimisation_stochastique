import numpy as np
from typing import Tuple

from objective_functions_numpy_streaming import (
    BaseObjectiveFunction,
    add_bias,
    add_bias_1d,
)


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear"

    def __call__(self, data: Tuple[np.ndarray, np.ndarray], theta: np.ndarray) -> np.ndarray:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X, y = data
        if self.bias:
            if X.ndim == 1:
                X = add_bias_1d(X)
            else:
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

    def grad(self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the linear regression loss, average over the batch
        """
        X, y = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, h)
        grad = np.dot(X.T, Y_pred - y) / n
        return grad

    def hessian(self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, average over the batch
        """
        X, _ = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        hessian = np.einsum("ni,nj->ij", X / n, X)
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
        hessian_col = np.dot(X.T, X[:, col]) / n
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
        Y_pred = np.dot(X, h)
        grad = np.dot(X.T, Y_pred - y) / n
        hessian_col = np.dot(X.T, X[:, col]) / n
        return grad, hessian_col

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss,
        average over the batch
        """
        X, y = data
        n = X.shape[0]
        if self.bias:
            X = add_bias(X)
        Y_pred = np.dot(X, h)
        grad = np.dot(X.T, Y_pred - y) / n
        hessian = np.einsum("ni,nj->ij", X / n, X)
        return grad, hessian

    def riccati(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the riccati term of the linear regression loss, works only for a single data point
        """
        X, _ = data
        n = X.shape[0]
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        riccati = X
        return riccati

    def grad_and_riccati(
        self, data: Tuple[np.ndarray, np.ndarray], h: np.ndarray, iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the riccati term of the linear regression loss, works only for a single data point
        """
        X, y = data
        n = X.shape[0]
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        Y_pred = np.dot(X, h)
        grad = (Y_pred - y) * X
        riccati = X
        return grad, riccati
