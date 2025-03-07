import numpy as np
from typing import Tuple

from objective_functions_numpy.streaming import BaseObjectiveFunction


class SphericalDistribution(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self):
        self.name = "Spherical Distribution"
        self.atol = 1e-6

    def __call__(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the objective function over a batch of data
        """
        X = data
        a = param[:-1]
        b = param[-1]
        if X.ndim == 1:
            return 0.5 * (np.linalg.norm(X - a) - b) ** 2
        else:
            return 0.5 * (np.linalg.norm(X - a, axis=1) - b) ** 2

    def get_param_dim(self, data: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        X = data
        return X.shape[-1] + 1

    def grad(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the objective function, average over the batch
        """
        X = data
        n = X.shape[0]
        a = param[:-1]
        b = param[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        grad = np.empty_like(param)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros(n), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad[:-1] = np.dot(diff.T, -1 + b * safe_inv_norm) / n
        grad[-1] = b - np.mean(norm)
        return grad

    def hessian(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X,
        average over the batch
        """
        X = data
        n, d = X.shape[0], param.size
        a = param[:-1]
        b = param[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros(n), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b * np.mean(safe_inv_norm)) * np.eye(d - 1) + np.einsum(
            "n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff
        )
        hessian[-1, :-1] = np.dot(diff.T, safe_inv_norm) / n
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return hessian

    def hessian_column(self, data: np.ndarray, param: np.ndarray, col: int) -> np.ndarray:
        """
        Compute a single column of the hessian of the objective function
        """
        X = data
        n = X.shape[0]
        a = param[:-1]
        b = param[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros(n), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        hessian_column = np.empty_like(param)
        if col < len(param) - 1:
            hessian_column[:-1] = np.dot(diff.T, b * safe_inv_norm**3 * diff[:, col]) / n
            hessian_column[col] += 1 - b * np.mean(safe_inv_norm)
            hessian_column[-1] = np.mean(diff[:, col] * safe_inv_norm)
        else:
            hessian_column[:-1] = np.dot(diff.T, safe_inv_norm) / n
            hessian_column[-1] = 1
        return hessian_column

    def grad_and_hessian(self, data: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute the grad and Hessian of the objective function,
        returns Id if h is close to X, average over the batch
        """
        X = data
        n, d = X.shape[0], param.size
        a = param[:-1]
        b = param[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros(n), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad_a = np.dot(diff.T, -1 + b * safe_inv_norm) / n
        grad_b = b - np.mean(norm)
        grad = np.append(grad_a, grad_b)
        hessian = np.empty((d, d))
        hessian[:-1, :-1] = (1 - b * np.mean(safe_inv_norm)) * np.eye(d - 1) + np.einsum(
            "n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff
        )
        hessian[-1, :-1] = np.dot(diff.T, safe_inv_norm) / n
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian

    def grad_and_hessian_column(
        self, data: np.ndarray, param: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a single column of the grad and Hessian of the objective function
        """
        X = data
        n = X.shape[0]
        a = param[:-1]
        b = param[-1]
        diff = X - a
        norm = np.linalg.norm(diff, axis=1)
        safe_inv_norm = np.where(
            np.isclose(norm, np.zeros(n), atol=self.atol),
            np.ones(n),
            1 / norm,
        )
        grad_a = np.dot(diff.T, -1 + b * safe_inv_norm) / n
        grad_b = b - np.mean(norm)
        grad = np.append(grad_a, grad_b)
        hessian_column = np.empty_like(param)
        if col < len(param) - 1:
            hessian_column[:-1] = np.dot(diff.T, b * safe_inv_norm**3 * diff[:, col]) / n
            hessian_column[col] += 1 - b * np.mean(safe_inv_norm)
            hessian_column[-1] = np.mean(diff[:, col] * safe_inv_norm)
        else:
            hessian_column[:-1] = np.dot(diff.T, safe_inv_norm) / n
            hessian_column[-1] = 1
        return grad, hessian_column
