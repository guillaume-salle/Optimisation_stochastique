import numpy as np
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction


class SphericalDistribution(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self):
        self.name = "Spherical Distribution"
        self.atol = 1e-7

    def __call__(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the objective function over a batch of data
        """
        X = data
        a = h[:-1]
        b = h[-1]
        if X.ndim == 1:
            return 0.5 * (np.linalg.norm(X - a) - b) ** 2
        else:
            return 0.5 * (np.linalg.norm(X - a, axis=1) - b) ** 2

    def get_theta_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data.squeeze()
        return X.shape[-1] + 1

    def grad(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function, works only for a single data point
        """
        X = data.squeeze()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        grad = np.empty_like(h)
        if norm < self.atol:
            grad[:-1] = 0
            grad[-1] = b
            return grad
        grad[:-1] = (-1 + b / norm) * diff
        grad[-1] = b - norm
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, works only for a single data point
        """
        X = data.squeeze()
        d = h.size
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian = np.eye(d)
            return hessian
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b / norm) * np.eye(d - 1) + (b / norm**3) * np.outer(
            diff, diff
        )
        hessian[-1, :-1] = diff / norm
        hessian[:-1, -1] = diff / norm
        hessian[-1, -1] = 1
        return hessian

    def hessian_column(self, data: np.ndarray, h: np.ndarray, col: int) -> np.ndarray:
        """
        Compute a single column of the objective function, works only for a single data point
        """
        X = data.squeeze()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            # Return the column of the identity matrix
            hessian_column = np.zeros_like(h)
            hessian_column[col] = 1
            return hessian_column
        hessian_column = np.empty_like(h)
        if col < h.size - 1:
            hessian_column[:-1] = (b / norm**3) * diff[col] * diff
            hessian_column[col] += 1 - b / norm
        else:
            hessian_column[:-1] = diff / norm
            hessian_column[-1] = 1
        return hessian_column

    def grad_and_hessian_column(
        self, data: np.ndarray, h: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the Hessian of the objective function,
        returns Id if h is close to X, works only for a single data point
        """
        X = data.squeeze()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        grad = np.empty_like(h)
        if norm < self.atol:
            grad[:-1] = 0
            grad[-1] = b
            hessian_column = np.zeros_like(h)
            hessian_column[col] = 1
            return grad, hessian_column
        grad[:-1] = (-1 + b / norm) * diff
        grad[-1] = b - norm
        hessian_column = np.empty_like(h)
        if col < h.size - 1:
            hessian_column[:-1] = (b / norm**3) * diff[col] * diff
            hessian_column[col] += 1 - b / norm
        else:
            hessian_column[:-1] = diff / norm
            hessian_column[-1] = 1
        return grad, hessian_column

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and Hessian of the objective function, works only for a single data point
        """
        X = data.squeeze()
        d = h.size
        a = h[:-1].copy()
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(X - a)
        grad = np.empty(d)
        if norm < self.atol:
            grad[:-1] = 0
            grad[-1] = b
            hessian = np.eye(d)
            return grad, hessian
        grad[:-1] = (-1 + b / norm) * diff
        grad[-1] = b - norm
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b / norm) * np.eye(d - 1) + (b / norm**3) * np.outer(
            diff, diff
        )
        hessian[-1, :-1] = diff / norm
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian
