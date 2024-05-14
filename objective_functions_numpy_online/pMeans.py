import numpy as np
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction


class pMeans(BaseObjectiveFunction):
    """
    p-means objective function
    """

    def __init__(self, p: float = 1.5):
        self.name = "p-means"
        self.p = p
        self.atol = 1e-7

    def __call__(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        return (np.linalg.norm(X - h, axis=1) ** self.p) / self.p

    def get_theta_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data
        return X.shape[-1]

    def grad(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function
        """
        X = data
        diff = h - X
        norm = np.linalg.norm(diff)
        grad = diff * norm ** (self.p - 2)
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.eye(d)
        else:
            return norm ** (self.p - 2) * (
                np.eye(d) - (2 - self.p) * np.outer(diff, diff) / norm**2
            )

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = data
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.zeros_like(h), np.eye(d)
        grad = diff * norm ** (self.p - 2)
        hessian = norm ** (self.p - 2) * (
            np.eye(d) - (2 - self.p) * np.outer(diff, diff) / norm**2
        )
        return grad, hessian
