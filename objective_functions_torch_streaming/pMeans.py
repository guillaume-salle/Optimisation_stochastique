import torch
from typing import Tuple

from objective_functions_torch_streaming import BaseObjectiveFunction


class pMeans(BaseObjectiveFunction):
    """
    p-means objective function
    """

    def __init__(self, p: float = 1.5):
        self.name = "p-means"
        self.p = p
        self.atol = 1e-6

    def __call__(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return (torch.norm(X - h, dim=1) ** self.p) / self.p

    def get_theta_dim(self, data: Tuple[torch.Tensor]) -> int:
        """
        Return the dimension of theta
        """
        X = data[0]
        return X.size(-1)

    def grad(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective function, average over the batch
        """
        X = data[0]
        n = X.size(0)
        diff = h - X
        norm = torch.norm(diff, dim=1)
        grad = torch.einsum("n,ni->i", norm ** (self.p - 2) / n, diff)
        return grad

    def hessian(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X,
        average over the batch
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
        # Divide by n here to have d+n operations instead of d^2
        hessian = torch.mean(norm ** (self.p - 2)) * torch.eye(d) - (
            2 - self.p
        ) * torch.einsum("n,ni,nj->ij", (safe_inv_norm**2) / n, diff, diff)
        return hessian

    def grad_and_hessian(
        self, data: Tuple[torch.Tensor], h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and Hessian of the objective function, returns Id if h is close to X,
        average over the batch
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
        grad = torch.einsum("n,ni->i", norm ** (self.p - 2) / n, diff)
        # Divide by n here to have d+n operations instead of d^2
        hessian = torch.eye(d) * torch.mean(norm ** (self.p - 2)) - (
            2 - self.p
        ) * torch.einsum("n,ni,nj->ij", safe_inv_norm ** (self.p - 4) / n, diff, diff)
        return grad, hessian
