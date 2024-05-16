import numpy as np
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction


class SphericalDistribution(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self):
        self.name = "Spherical Distribution"
        self.atol = 1e-6

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
        X = data
        return X.shape[-1] + 1

    def grad(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function, average over the batch
        """
        X = data
        n = X.shape[0]
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        grad_a = np.dot(diff.T, -1 + b * safe_inv_norm) / n
        grad_b = b - np.mean(norm)
        grad = np.append(grad_a, grad_b)
        return grad

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X,
        average over the batch
        """
        X = data
        n, d = X.shape[0], h.size
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b * np.mean(safe_inv_norm)) * np.eye(
            d - 1
        ) + np.einsum("n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff)
        hessian[-1, :-1] = np.dot(diff.T, safe_inv_norm) / n
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return hessian

    def grad_and_hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the grad and Hessian of the objective function,
        returns Id if h is close to X, average over the batch
        """
        X = data
        n, d = X.shape[0], h.size
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones_like(norm),
            1 / norm,
        )
        grad_a = np.dot(diff.T, -1 + b * safe_inv_norm) / n
        grad_b = b - np.mean(norm)
        grad = np.append(grad_a, grad_b)
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b * np.mean(safe_inv_norm)) * np.eye(
            d - 1
        ) + np.einsum("n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff)
        hessian[-1, :-1] = np.dot(diff.T, safe_inv_norm) / n
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian
