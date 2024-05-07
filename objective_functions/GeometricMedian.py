import numpy as np
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction


class GeometricMedian(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self):
        self.name = "Geometric median"

    def __call__(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.linalg.norm(X - h) - np.linalg.norm(X)

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1]

    def grad(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        diff = X - h
        norm = np.linalg.norm(diff)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros_like(h)
        else:
            return -diff / norm

    def hessian(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        diff = X - theta
        norm = np.linalg.norm(diff)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros((theta.shape[0], theta.shape[0]))
        else:
            return np.eye(theta.shape[0]) / norm - np.outer(diff, diff) / norm**3

    def grad_and_hessian(
        self, X: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = 1e-8
        diff = X - h
        norm = np.linalg.norm(diff)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros_like(h), np.zeros((h.shape[0], h.shape[0]))
        else:
            grad = -diff / norm
            hessian = np.eye(h.shape[0]) / norm - np.outer(diff, diff) / norm**3
            return grad, hessian

    def grad_and_riccati(
        self, X: np.ndarray, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Compute gradient
        epsilon = 1e-8
        diff = X - h
        norm = np.linalg.norm(X - h)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros_like(h), np.zeros_like(h)
        grad = -diff / norm

        # Compute Riccati
        Z = np.random.randn(h.shape[0])
        alpha = 1 / (iter * np.log(iter + 1))
        riccati = (self.grad(X, h + alpha * Z) - grad) * np.sqrt(norm) / alpha
        # ??? Not in article
        # Donne environ H * Z * norm si alpha est petit, donc E[phi phi.T] = H^2 * norm^2 ??

        return grad, riccati
