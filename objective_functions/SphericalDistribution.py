import torch
from typing import Tuple

from objective_functions import BaseObjectiveFunction


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

    def get_theta_dim(self, X: torch.Tensor) -> int:
        """
        Return the dimension of theta
        """
        return X.size(-1) + 1

    def grad(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the objective function, average over the batch
        """
        n = X.size(0)
        a = h[:-1].copy()
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
        return torch.cat([grad_a, grad_b])

    def hessian(self, X: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the objective function, returns Id if h is close to X, average over the batch
        """
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
        self, X: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and Hessian of the objective function
        """
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
        grad = torch.cat([grad_a, grad_b])
        hessian = torch.empty(d + 1, d + 1, device=h.device, dtype=h.dtype)
        hessian[:-1, :-1] = (1 - b * torch.mean(safe_inv_norm)) * torch.eye(
            d
        ) + torch.einsum("n,ni,nj->ij", b * safe_inv_norm**3 / n, diff, diff)
        hessian[-1, :-1] = torch.einsum("n, ni->i", safe_inv_norm / n, diff)
        hessian[:-1, -1] = hessian[-1, :-1]
        hessian[-1, -1] = 1
        return grad, hessian

        # if norm == 0:
        #     return np.array([0, b]), None
        # else:
        #     grad = np.concatenate([a - X + b * (X - a) / norm, np.array([b - norm])])
        #     # eye_part = (1 - b / norm) * np.eye(len(a)) + (b / norm**3) * np.outer(
        #     #     X - a, X - a
        #     # )
        #     # matrix = (X-a) @ (X-a).T
        #     # assert matrix.shape == (len(a), len(a))
        #     # eye_part = (1 - b / norm) * np.eye(len(a)) + (b / norm**3) * matrix
        #     # vector_part = (X - a)[:, np.newaxis] / norm
        #     # scalar_part = np.array([[1]])
        #     # top_right = vector_part
        #     # bottom_left = vector_part.T

        #     # hessian = np.block([[eye_part, top_right], [bottom_left, scalar_part]])
        #     # return grad, hessian

        #     hessian = np.zeros((len(a) + 1, len(a) + 1))
        #     matrix = np.outer(X - a, X - a)
        #     assert matrix.shape == (len(a), len(a))
        #     hessian[:-1, :-1] = (1 - b / norm) * np.eye(len(a)) + b / norm**3 * matrix
        #     hessian[-1, :-1] = (X - a) / norm
        #     hessian[:-1, -1] = (X - a) / norm
        #     hessian[-1, -1] = 1
        #     return grad, hessian
