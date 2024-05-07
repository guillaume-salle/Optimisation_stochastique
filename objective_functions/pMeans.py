import numpy as np
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction


class pMeans(BaseObjectiveFunction):
    """
    p-means objective function
    """

    def __init__(self, p: float = 1.5):
        self.name = "p-means"
        self.p = p

    def __call__(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(X - h) ** self.p) / self.p

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1]

    def grad(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        return -(X - h) * np.linalg.norm(X - h) ** (self.p - 2)

    def hessian(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        epsilon = 1e-8
        diff = X - h
        norm = np.linalg.norm(diff)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros((h.shape[0], h.shape[0]))
        else:
            return norm ** (self.p - 2) * (
                np.eye(h.shape[0]) - (2 - self.p) * np.outer(diff, diff) / norm**2
            )

    def grad_and_hessian(
        self, X: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        epsilon = 1e-8
        diff = X - h
        norm = np.linalg.norm(diff)
        if np.isclose(norm, 0, atol=epsilon):
            return np.zeros_like(h), np.zeros((h.shape[0], h.shape[0]))
        grad = -diff * norm ** (self.p - 2)
        hessian = norm ** (self.p - 2) * (
            np.eye(h.shape[0]) - (2 - self.p) * np.outer(diff, diff) / norm**2
        )
        return grad, hessian
