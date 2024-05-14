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
        X = data.atleast_2d()
        a = h[:-1]
        b = h[-1]
        return 0.5 * (np.linalg.norm(X - a, axis=1) - b) ** 2

    def get_theta_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data
        return X.shape[-1] + 1

    def grad(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function
        """
        X = data
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        grad_a = np.zeros_like(a) if norm < self.atol else (-1 + b / norm) * diff
        grad_b = b - norm
        grad = np.append(grad_a, grad_b)
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function
        """
        X = data
        d = h.size
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian = np.eye(d)
            return hessian
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b / norm) * np.eye(d - 1) + b / norm**3 * np.outer(
            diff, diff
        )
        hessian[-1, :-1] = diff / norm
        hessian[:-1, -1] = diff / norm
        hessian[-1, -1] = 1
        return hessian

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = data
        d = h.size
        a = h[:-1].copy()
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(X - a)
        grad_a = np.zeros_like(a) if norm < self.atol else (-1 + b / norm) * diff
        grad_b = b - norm
        grad = np.append(grad_a, grad_b)
        if norm < self.atol:
            hessian = np.eye(d)
            return grad, hessian
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b / norm) * np.eye(d - 1) + b / norm**3 * np.outer(
            diff, diff
        )
        hessian[-1, :-1] = diff / norm
        hessian[:-1, -1] = diff / norm
        hessian[-1, -1] = 1
        return grad, hessian
