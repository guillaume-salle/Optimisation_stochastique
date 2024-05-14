import torch
from typing import Tuple

from objective_functions_torch_streaming import BaseObjectiveFunction


class SphericalDistribution(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self):
        self.name = "Spherical Distribution"
        self.atol = 1e-6

    def __call__(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        a = h[:-1]
        b = h[-1]
        return 0.5 * (torch.norm(X - a, dim=1) - b) ** 2

    def get_theta_dim(self, data: Tuple[torch.Tensor]) -> int:
        """
        Return the dimension of theta
        """
        X = data[0]
        return X.size(-1) + 1

    def grad(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective function, average over the batch
        """
        X = data[0]
        n = X.size(0)
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad_a = torch.einsum("n,ni->i", (-1 + b * safe_inv_norm) / n, diff)
        grad_b = b - torch.mean(norm)
        return torch.cat([grad_a, grad_b.unsqueeze(0)])

    def hessian(self, data: Tuple[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X, average over the batch
        """
        X = data[0]
        n, d = X.size()
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        hessian = torch.empty(d + 1, d + 1, device=h.device, dtype=h.dtype)
        hessian[:-1, :-1] = (1 - b * torch.mean(safe_inv_norm)) * torch.eye(
            d
        ) + torch.einsum("n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff)
        hessian[-1, :-1] = torch.einsum("n, ni->i", safe_inv_norm / n, diff)
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
        a = h[:-1]
        b = h[-1]
        diff = X - a
        norm = torch.norm(diff, dim=1)
        safe_inv_norm = torch.where(
            torch.isclose(norm, torch.zeros_like(norm), atol=self.atol),
            torch.ones_like(norm),
            1 / norm,
        )
        grad_a = torch.einsum("n,ni->i", (-1 + b * safe_inv_norm) / n, diff)
        grad_b = b - torch.mean(norm)
        grad = torch.cat([grad_a, grad_b.unsqueeze(0)])
        hessian = torch.empty(d + 1, d + 1, device=h.device, dtype=h.dtype)
        hessian[:-1, :-1] = (1 - b * torch.mean(safe_inv_norm)) * torch.eye(
            d
        ) + torch.einsum("n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff)
        hessian[-1, :-1] = torch.einsum("n, ni->i", safe_inv_norm / n, diff)
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian
