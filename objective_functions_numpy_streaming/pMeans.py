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
        self.atol = 1e-6
        if p < 1:
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
        Compute the gradient of the objective function, average over the batch
        """
        X = data
        n = X.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        if self.p < 2:
            safe_norm = np.where(
                np.isclose(norm, np.zeros(n), atol=self.atol),
                np.ones(n),
                norm,
            )
        else:
            safe_norm = norm
        grad = np.dot(diff.T, safe_norm ** (self.p - 2) / n)
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
        if self.p < 4:
            safe_norm = np.where(
                np.isclose(norm, np.zeros(n), atol=self.atol),
                np.ones(n),
                norm,
            )
        else:
            safe_norm = norm
        # Divide by n here to have 2n operations instead of d^2
        hessian = np.mean(safe_norm ** (self.p - 2)) * np.eye(d) + np.einsum(
            "n,ni,nj->ij", -(2 - self.p) * safe_norm ** (self.p - 4) / n, diff, diff
        )
        return hessian

    def hessian_column(self, data: np.ndarray, h: np.ndarray, col: int) -> np.ndarray:
        """
        Compute a single column of the hessian of the objective function
        """
        X = data
        n = X.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        if self.p < 4:
            safe_norm = np.where(
                np.isclose(norm, np.zeros(n), atol=self.atol),
                np.ones(n),
                norm,
            )
        else:
            safe_norm = norm
        hessian_col = np.dot(
            diff.T, -(2 - self.p) * safe_norm ** (self.p - 4) * diff[:, col] / n
        )
        hessian_col[col] += np.mean(safe_norm ** (self.p - 2))
        return hessian_col

    def grad_and_hessian_column(
        self, data: np.ndarray, h: np.ndarray, col: int
    ) -> Tuple[np.ndarray]:
        """
        Compute the gradient and a single column of the Hessian of the objective function
        """
        X = data
        n = X.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        if self.p < 4:
            safe_norm = np.where(
                np.isclose(norm, np.zeros(n), atol=self.atol),
                np.ones(n),
                norm,
            )
        else:
            safe_norm = norm
        grad = np.dot(diff.T, safe_norm ** (self.p - 2) / n)
        hessian_col = np.dot(
            diff.T, -(2 - self.p) * safe_norm ** (self.p - 4) * diff[:, col] / n
        )
        hessian_col[col] += np.mean(safe_norm ** (self.p - 2))
        return grad, hessian_col

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
