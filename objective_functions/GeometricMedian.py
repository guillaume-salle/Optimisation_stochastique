import torch
import math
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction


class GeometricMedian(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self):
        self.name = "Geometric median"
        self.atol = 1e-6

    def __call__(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        X = torch.atleast_2d(X)
        return torch.norm(X - h, dim=1) - torch.norm(X, dim=1)

    def get_theta_dim(self, X: torch.Tensor) -> int:
        """
        Return the dimension of theta
        """
        return X.size(-1)

    def grad(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective function, returns 0 if h is close to X
        sum over the batch and normalize by the batch size
        """
        n = X.size(0)
        diff = h - X
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad = torch.einsum("n,ni->i", safe_inv_norm, diff)
        return grad / n

    def hessian(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        sum over the batch and normalize by the batch size
        """
        n, d = X.size
        diff = h - X
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        # Divide here by n to have d+n operations instead of d^2
        hessian = torch.eye(d) * torch.mean(safe_inv_norm) - torch.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return hessian

    def grad_and_hessian(
        self, X: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d = X.size
        diff = h - X
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad = torch.einsum("n,ni->i", safe_inv_norm, diff)
        hessian = torch.eye(d) * torch.mean(safe_inv_norm) - torch.einsum(
            "n,ni,nj->ij", safe_inv_norm**3 / n, diff, diff
        )
        return grad / n, hessian

    def grad_and_riccati(
        self, X: torch.Tensor, h: torch.Tensor, iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d = X.size()
        if n != 1:
            raise ValueError("Riccati is only implemented for batch size 1")
        diff = h - X
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad = safe_inv_norm * diff
        Z = torch.randn(d)
        alpha = 1 / (iter * math.log(iter + 1))
        grad_Z = self.grad(X, h + alpha * Z)
        riccati = (grad_Z - grad) * torch.sqrt(norm) / alpha
        return grad, riccati

    def grad_batch(
        self, X: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient of the objective function, returns 0 if h is close to X
        do not sum over the batch
        """
        diff = h - X
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad_all = safe_inv_norm[:, None] * diff
        return grad_all, norm

    def grad_and_riccati_batch(
        self, X: torch.Tensor, h: torch.Tensor, iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d = X.size()
        grad_all, norm = self.grad_all(X, h)
        grad = grad_all.mean(dim=0)
        Z = torch.randn(n, d)
        alpha = 1 / (iter * math.log(iter + 1))
        grad_Z_all, _ = self.grad_all(X, h + alpha * Z)
        riccati = (grad_Z_all - grad_all) * torch.sqrt(norm)[:, None] / alpha
        return grad, riccati.t()
