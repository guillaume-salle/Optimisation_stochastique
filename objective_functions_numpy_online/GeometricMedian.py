import numpy as np
import math
from typing import Tuple

from objective_functions_numpy_online import BaseObjectiveFunction


class GeometricMedian(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self):
        self.name = "Geometric median"
        self.atol = 1e-7

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
        """
        X = data.squeeze()
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.zeros_like(h)
        else:
            return diff / norm

    def hessian(self, data: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.eye(d)
        else:
            return np.eye(d) / norm - np.outer(diff, diff / (norm**3))

    def hessian_column(
        self, data: np.ndarray, h: np.ndarray, column: int
    ) -> np.ndarray:
        """
        Compute a single column of the Hessian of the objective function
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian_col = np.zeros(d)
            hessian_col[column] = 1
            return hessian_col
        else:
            hessian_col = (-diff[column] / (norm**3)) * diff
            hessian_col[column] += 1 / norm
            return hessian_col

    def grad_and_hessian(
        self, data: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.zeros_like(h), np.eye(d)
        else:
            grad = diff / norm
            hessian = np.eye(d) / norm - np.outer(diff, diff / (norm**3))
            return grad, hessian

    def grad_and_hessian_column(
        self, data: np.ndarray, h: np.ndarray, column: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the Hessian of the objective function
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian_col = np.zeros(d)
            hessian_col[column] = 1
            return np.zeros_like(h), hessian_col
        else:
            grad = diff / norm
            hessian_col = (-diff[column] / (norm**3)) * diff
            hessian_col[column] += 1 / norm
            return grad, hessian_col

    def riccati(
        self, data: np.ndarray, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the riccati term of the objective function
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the Riccati term,
            # so the outer product averages to the identity matrix
            z = np.random.randint(0, d)
            riccati = np.zeros_like(h)
            riccati[z] = math.sqrt(d)
            return np.zeros_like(h), riccati
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (iter * math.log(iter + 1))
        riccati = (self.grad(X, h + alpha * Z) - grad) * np.sqrt(norm) / alpha
        return riccati

    def grad_and_riccati(
        self, data: np.ndarray, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and riccati term of the objective function
        """
        X = data.squeeze()
        d = h.shape[0]
        diff = h - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the Riccati term,
            # so the outer product averages to the identity matrix
            z = np.random.randint(0, d)
            riccati = np.zeros_like(h)
            riccati[z] = math.sqrt(d)
            return np.zeros_like(h), riccati
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (iter * math.log(iter + 1))
        riccati = (self.grad(X, h + alpha * Z) - grad) * np.sqrt(norm) / alpha
        return grad, riccati
