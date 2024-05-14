import torch
import numpy as np
from typing import Tuple

from objective_functions_torch_streaming import BaseObjectiveFunction


class SphericalDistributionNumpyOnline(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self):
        self.name = "Spherical Distribution"
        self.atol = 1e-6

    def __call__(self, X: torch.Tensor, h: torch.Tensor) -> np.ndarray:
        X, h = X.numpy(), h.numpy()
        a = h[:-1]
        b = h[-1]
        return 0.5 * (np.linalg.norm(X - a, axis=1) - b) ** 2

    def get_theta_dim(self, data: Tuple[torch.Tensor]) -> int:
        """
        Return the dimension of theta
        """
        X = data[0]
        return X.size(-1) + 1

    def grad(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> np.ndarray:
        """
        Compute the gradient of the objective function
        """
        X = data[0]
        n = X.size(0)
        if n != 1:
            raise ValueError(
                "Online version of the gradient is only available for batch size 1"
            )
        X, h = X.numpy().squeeze(), h.numpy()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        safe_inv_norm = 1 / norm if norm > self.atol else 1
        grad_a = (-1 + b * safe_inv_norm) * diff
        grad_b = b - norm
        return np.concatenate([grad_a, np.expand_dims(grad_b, axis=0)])

    def hessian(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> np.ndarray:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        """
        X = data[0]
        n, d = X.size()
        if n != 1:
            raise ValueError(
                "Online version of the gradient is only available for batch size 1"
            )
        X, h = X.numpy().squeeze(), h.numpy()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        safe_inv_norm = 1 / norm if norm > self.atol else 1
        hessian = np.empty(d + 1, d + 1, dtype=h.dtype)
        hessian[:-1, :-1] = (1 - b * safe_inv_norm) * np.eye(
            d
        ) + b * safe_inv_norm**3 * np.outer(diff, diff)
        hessian[-1, :-1] = safe_inv_norm * diff
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return hessian

    def grad_and_hessian(
        self, data: Tuple[torch.Tensor], h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and Hessian of the objective function
        """
        X = data[0]
        n, d = X.size()
        if n != 1:
            raise ValueError(
                "Online version of the gradient is only available for batch size 1"
            )
        X, h = X.numpy().squeeze(), h.numpy()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = np.linalg.norm(diff)
        safe_inv_norm = 1 / norm if norm > self.atol else 1
        grad_a = (-1 + b * safe_inv_norm) * diff
        grad_b = b - norm
        grad = np.concatenate([grad_a, np.expand_dims(grad_b, axis=0)])
        hessian = np.empty(d + 1, d + 1, dtype=h.dtype)
        hessian[:-1, :-1] = (1 - b * safe_inv_norm) * np.eye(
            d
        ) + d * safe_inv_norm**3 * np.outer(diff, diff)
        hessian[-1, :-1] = safe_inv_norm * diff
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian
