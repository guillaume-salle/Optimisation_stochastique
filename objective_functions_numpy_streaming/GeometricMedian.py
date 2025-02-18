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

    def __call__(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the objective function, works for a single point or a batch of points
        """
        X = data
        if X.ndim == 1:
            return np.linalg.norm(X - param) - np.linalg.norm(X)
        else:
            return np.linalg.norm(X - param, axis=1) - np.linalg.norm(X, axis=1)

    def get_param_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data
        return X.shape[-1]

    def grad(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function, returns 0 if h is close to X
        average over the batch
        """
        X = data
        n = X.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad = np.dot(diff.T, safe_inv_norm) / n
        return grad

    def hessian(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        average over the batch
        """
        X = data
        n, d = X.shape
        diff = param - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        # Divide here by n to have d+n operations instead of d^2
        hessian = np.eye(d) * np.mean(safe_inv_norm) - np.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return hessian

    def hessian_column(self, data: np.ndarray, param: np.ndarray, col: int) -> np.ndarray:
        """
        Compute a single column of the hessian of the objective function
        """
        X = data
        n = X.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        hessian_column = np.dot(diff.T, -(safe_inv_norm**3) * diff[:, col]) / n
        hessian_column[col] += np.mean(safe_inv_norm)
        return hessian_column

    def grad_and_hessian_column(
        self, data: np.ndarray, param: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the hessian of the objective function
        """
        X = data
        n = X.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad = np.dot(diff.T, safe_inv_norm) / n
        hessian_column = np.dot(diff.T, -(safe_inv_norm**3) * diff[:, col]) / n
        hessian_column[col] += np.mean(safe_inv_norm)
        return grad, hessian_column

    def grad_and_hessian(
        self, data: np.ndarray, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and Hessian of the objective function, (0,Id) if h is close to X,
        average over the batch
        """
        X = data
        n, d = X.shape
        diff = param - X
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros_like(norm), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad = np.dot(diff.T, safe_inv_norm) / n
        hessian = np.eye(d) * np.mean(safe_inv_norm) - np.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return grad, hessian

    def sherman_morrison(
        self, data: np.ndarray, param: np.ndarray, n_iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Sherman-Morrison term of the objective function
        """
        X = data
        n, d = X.shape
        if n != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the Sherman-Morrison term,
            # so the outer product average to the identity matrix
            grad = np.zeros(d)
            z = random.randint(0, d - 1)
            riccati = np.zeros(d)
            riccati[z] = 1
            return grad, riccati
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (n_iter * math.log(n_iter + 1))
        riccati = (self.grad(X, param + alpha * Z) - grad) * math.sqrt(norm) / alpha
        return riccati.squeeze()

    def grad_and_sherman_morrison(
        self, data: np.ndarray, param: np.ndarray, n_iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and Sherman_Morrison term of the objective function
        """
        X = data
        n, d = X.shape
        if n != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the Sherman-Morrison term,
            # so the outer product average to the identity matrix
            grad = np.zeros(d)
            z = random.randint(0, d - 1)
            riccati = np.zeros(d)
            riccati[z] = 1
            return grad, riccati
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (n_iter * math.log(n_iter + 1))
        riccati = (self.grad(X, param + alpha * Z) - grad) * math.sqrt(norm) / alpha
        return grad.squeeze(), riccati.squeeze()
