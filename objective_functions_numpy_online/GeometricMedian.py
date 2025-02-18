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
        """
        X = data.squeeze()
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.zeros_like(param)
        else:
            return diff / norm

    def hessian(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data.squeeze()
        d = param.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.eye(d)
        else:
            return np.eye(d) / norm - np.outer(diff, diff / (norm**3))

    def hessian_column(self, data: np.ndarray, param: np.ndarray, col: int) -> np.ndarray:
        """
        Compute a single column of the Hessian of the objective function
        """
        X = data.squeeze()
        d = param.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian_col = np.zeros(d)
            hessian_col[col] = 1
            return hessian_col
        else:
            hessian_col = (-diff[col] / (norm**3)) * diff
            hessian_col[col] += 1 / norm
            return hessian_col

    def grad_and_hessian(
        self, data: np.ndarray, param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the objective function
        """
        X = data.squeeze()
        d = param.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            return np.zeros_like(param), np.eye(d)
        else:
            grad = diff / norm
            hessian = np.eye(d) / norm - np.outer(diff, diff / (norm**3))
            return grad, hessian

    def grad_and_hessian_column(
        self, data: np.ndarray, param: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the Hessian of the objective function
        """
        X = data.squeeze()
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:
            hessian_col = np.zeros_like(param)
            hessian_col[col] = 1
            return np.zeros_like(param), hessian_col
        else:
            grad = diff / norm
            hessian_col = (-diff[col] / (norm**3)) * diff
            hessian_col[col] += 1 / norm
            return grad, hessian_col

    def sherman_morrison(
        self, data: np.ndarray, param: np.ndarray, n_iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Sherman-Morrison term of the objective function
        """
        X = data.squeeze()
        d = param.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the sherman_morrison term,
            # so the outer product averages to the identity matrix
            z = np.random.randint(0, d)
            sherman_morrison = np.zeros_like(param)
            sherman_morrison[z] = math.sqrt(d)
            return np.zeros_like(param), sherman_morrison
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (n_iter * math.log(n_iter + 1))
        sherman_morrison = (self.grad(X, param + alpha * Z) - grad) * np.sqrt(norm) / alpha
        return sherman_morrison

    def grad_and_sherman_morrison(
        self, data: np.ndarray, param: np.ndarray, n_iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and Sherman_Morrison term of the objective function
        """
        X = data.squeeze()
        d = param.shape[0]
        diff = param - X
        norm = np.linalg.norm(diff)
        if norm < self.atol:  # Randomly select a direction for the sherman_morrison term,
            # so the outer product averages to the identity matrix
            z = np.random.randint(0, d)
            sherman_morrison = np.zeros_like(param)
            sherman_morrison[z] = math.sqrt(d)
            return np.zeros_like(param), sherman_morrison
        grad = diff / norm
        Z = np.random.randn(d)
        alpha = 1 / (n_iter * math.log(n_iter + 1))
        sherman_morrison = (self.grad(X, param + alpha * Z) - grad) * np.sqrt(norm) / alpha
        return grad, sherman_morrison
