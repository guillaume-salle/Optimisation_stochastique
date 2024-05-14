import torch
import math
from typing import Any, Tuple

from objective_functions_torch_streaming import BaseObjectiveFunction


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

    def get_theta_dim(self, data: Tuple[torch.Tensor]) -> int:
        """
        Return the dimension of theta
        """
        X = data[0]
        return X.size(-1)

    def grad(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective function, returns 0 if h is close to X
        sum over the batch and normalize by the batch size
        """
        X = data[0]
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

    def hessian(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X
        sum over the batch and normalize by the batch size
        """
        X = data[0]
        n, d = X.size()
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
        self, data: Tuple[torch.Tensor], h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and Hessian of the objective function, average over the batch
        returns (0,Id) if h is close to X
        """
        X = data[0]
        n, d = X.size()
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
        self, data: Tuple[torch.Tensor], h: torch.Tensor, iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and riccati term of the objective function,
        works only for batch size 1
        """
        X = data[0]
        n, d = X.size()
        if n != 1:
            raise ValueError("Riccati is only implemented for batch size 1")
        X = X.squeeze()
        diff = h - X
        norm = torch.norm(diff)
        if norm < self.atol:
            return torch.zeros_like(h), torch.zeros_like(h)
        grad = diff / norm
        Z = torch.randn(d)
        alpha = 1 / (iter * math.log(iter + 1))
        # grad expects a tuple batch
        grad_Z = self.grad([X.unsqueeze(0)], h + alpha * Z)
        riccati = (grad_Z - grad) * torch.sqrt(norm) / alpha
        return grad, riccati

    # def grad_batch(
    #     self, data: Tuple[torch.Tensor], h: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute the gradient of the objective function, returns 0 if h is close to X
    #     do not sum over the batch
    #     """
    #     diff = h - X
    #     norm = torch.norm(diff, dim=1)
    #     safe_inv_norm = torch.where(
    #         torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
    #         torch.ones_like(norm),
    #         1 / norm,
    #     )
    #     grad_all = safe_inv_norm[:, None] * diff
    #     return grad_all, norm

    # def grad_and_riccati_batch(
    #     self, data: Tuple[torch.Tensor], h: torch.Tensor, iter: int
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute the gradient and riccati term of the objective function,
    #     average the gradient over the batch and returns all the ricatti terms (n,d)
    #     """
    #     X = data[0]
    #     n, d = X.size()
    #     grad_all, norm = self.grad_all(X, h)
    #     grad = grad_all.mean(dim=0)
    #     Z = torch.randn(n, d)
    #     alpha = 1 / (iter * math.log(iter + 1))
    #     grad_Z_all, _ = self.grad_all(X, h + alpha * Z)
    #     riccati = (grad_Z_all - grad_all) * torch.sqrt(norm)[:, None] / alpha
    #     return grad, riccati.t()
