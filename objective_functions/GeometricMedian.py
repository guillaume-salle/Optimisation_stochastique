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
        norm = np.linalg.norm(X - h)
        if norm == 0:
            return np.zeros_like(h)
        else:
            return -(X - h) / norm

    def hessian(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X - theta)
        if norm == 0:
            return np.zeros((theta.shape[0], theta.shape[0]))
        else:
            return (
                np.eye(theta.shape[0]) - np.outer(X - theta, X - theta) / norm**2
            ) / norm

    def grad_and_hessian(
        self, X: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.linalg.norm(X - h)
        if norm == 0:
            return np.zeros_like(h), np.zeros((h.shape[0], h.shape[0]))
        else:
            grad = -(X - h) / norm
            hessian = (np.eye(h.shape[0]) - np.outer(X - h, X - h) / norm**2) / norm
            return grad, hessian

    def grad_and_riccati(
        self, X: np.ndarray, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad = self.grad(X, h)
        Z = np.random.randn(h.shape[0])
        alpha = 1 / (iter * np.log(iter + 1))
        riccati = self.grad(X, h + alpha * Z) - grad
        return grad, riccati
