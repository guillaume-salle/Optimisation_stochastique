import numpy as np
from typing import Tuple

from objective_functions_numpy_streaming import BaseObjectiveFunction


class pMeans(BaseObjectiveFunction):
    """
    p-means objective function
    """

    def __init__(self, p: float = 1.5):
        self.name = "p-means"
        self.p = p
        if p < 4 and p >= 1:
            self.hessian = self.hessian_with_inv
            self.grad_and_hessian = self.grad_and_hessian_with_inv
        elif p < 1:
            raise ValueError(
                "The p-means objective function is only defined for p >= 1"
            )
        self.atol = 1e-6

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
        Compute the gradient of the objective function, average over the batch
        """
        X = data
        n = X.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        grad = np.dot(norm ** (self.p - 2) / n, diff)
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X,
        average over the batch. This is the case when p >= 4, we don't have to invert the norm
        """
        X = data
        n, d = X.shape
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        # Divide by n here to have 2n operations instead of d^2
        hessian = np.mean(norm ** (self.p - 2)) * np.eye(d) - (2 - self.p) * np.einsum(
            "n,ni,nj->ij", norm ** (self.p - 4) / n, diff, diff
        )
        return hessian

    def hessian_with_inv(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X,
        average over the batch. This is the case when p < 4, we have to invert the norm
        """
        X = data
        n, d = X.shape
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        # Divide by n here to have 2n operations instead of d^2
        hessian = np.mean(norm ** (self.p - 2)) * np.eye(d) - (2 - self.p) * np.einsum(
            "n,ni,nj->ij", safe_inv_norm ** (4 - self.p) / n, diff, diff
        )
        return hessian

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function,
        returns Id if h is close to X, average over the batch.
        This is the case when p >= 4, we don't have to invert the norm
        """
        X = data
        n, d = X.shape
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        grad = np.dot(norm ** (self.p - 2) / n, diff)
        hessian = np.mean(norm ** (self.p - 2)) * np.eye(d) - (2 - self.p) * np.einsum(
            "n,ni,nj->ij", norm ** (self.p - 4) / n, diff, diff
        )
        return grad, hessian

    def grad_and_hessian_with_inv(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function,
        returns Id if h is close to X, average over the batch.
        This is the case when p < 4, we have to invert the norm
        """
        X = data
        n, d = X.shape
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        grad = np.dot(norm ** (self.p - 2) / n, diff)
        hessian = np.mean(norm ** (self.p - 2)) * np.eye(d) - (2 - self.p) * np.einsum(
            "n,ni,nj->ij", safe_inv_norm ** (4 - self.p) / n, diff, diff
        )
        return grad, hessian
