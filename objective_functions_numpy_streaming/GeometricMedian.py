import numpy as np
import math
import random
from typing import Tuple

from objective_functions_numpy_streaming import BaseObjectiveFunction


class GeometricMedian(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self):
        self.name = "Geometric median"
        self.atol = 1e-6

    def __call__(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the objective function, works for a single point or a batch of points
        """
        X = data
        if X.ndim == 1:
            return np.linalg.norm(X - h) - np.linalg.norm(X)
        else:
            return np.linalg.norm(X - h, axis=1) - np.linalg.norm(X, axis=1)

    def get_theta_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data
        return X.shape[-1]

    def grad(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function, returns 0 if h is close to X
        average over the batch
        """
        X = data
        n = X.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        grad = np.dot(diff.T, safe_inv_norm) / n
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        average over the batch
        """
        X = data
        n, d = X.shape
        diff = h - X
        norm = np.linalg.norm(diff, dim=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        # Divide here by n to have d+n operations instead of d^2
        hessian = np.eye(d) * np.mean(safe_inv_norm) - np.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return hessian

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and Hessian of the objective function, (0,Id) if h is close to X,
        average over the batch
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
        grad = np.dot(diff.T, safe_inv_norm) / n
        hessian = np.eye(d) * np.mean(safe_inv_norm) - np.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return grad, hessian

    def grad_and_riccati(
        self, data: np.ndarray, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and riccati term of the objective function
        """
        X = data
        n, d = X.shape
        if n != 1:
            raise ValueError("Riccati is only implemented for batch size 1")
        X = X.squeeze()
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            # Randomly select a direction for the Riccati term, so the outer product
            # average to the identity matrix
            z = random.randint(0, d - 1)
            riccati = np.zeros(d)
            riccati[z] = 1
            return np.zeros_like(h), riccati
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (iter * math.log(iter + 1))
        riccati = (self.grad(X, h + alpha * Z) - grad) * np.sqrt(norm) / alpha
        return grad, riccati
