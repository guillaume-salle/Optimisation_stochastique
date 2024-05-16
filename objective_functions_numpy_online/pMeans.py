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
        if p >= 4:
            self.hessian = self.hessian_without_inv
            self.grad_and_hessian = self.grad_and_hessian_without_inv
        elif p < 4 and p >= 1:
            self.hessian = self.hessian_with_inv
            self.grad_and_hessian = self.grad_and_hessian_with_inv
        else:
            raise ValueError(
                "The p-means objective function is only defined for p >= 1"
            )

    def __call__(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the objective function over a batch of data or a single data point
        """
        X = data
        if X.ndim == 1:
            return (np.linalg.norm(X - h) ** self.p) / self.p
        else:
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
        X = data.squeeze()
        diff = h - X
        norm = np.linalg.norm(diff)
        grad = diff * norm ** (self.p - 2)
        return grad

    def hessian_without_inv(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        hessian = norm ** (self.p - 2) * np.eye(d) - (2 - self.p) * norm ** (
            self.p - 4
        ) * np.outer(diff, diff)
        return hessian

    def hessian_with_inv(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian = np.eye(d)
        else:
            hessian = norm ** (self.p - 2) * np.eye(d) - (2 - self.p) * norm ** (
                self.p - 4
            ) * np.outer(diff, diff)
        return hessian

    def grad_and_hessian_without_inv(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        grad = diff * norm ** (self.p - 2)
        hessian = norm ** (self.p - 2) * np.eye(d) - (2 - self.p) * norm ** (
            self.p - 4
        ) * np.outer(diff, diff)
        return grad, hessian

    def grad_and_hessian_with_inv(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            grad = np.zeros_like(h)
            hessian = np.eye(d)
            return grad, hessian
        grad = diff * norm ** (self.p - 2)
        hessian = norm ** (self.p - 2) * np.eye(d) - (2 - self.p) * norm ** (
            self.p - 4
        ) * np.outer(diff, diff)
        return grad, hessian
