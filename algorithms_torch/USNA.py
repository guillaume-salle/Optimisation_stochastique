import torch
import random
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Set to 1.0 in the article
        c_alpha: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, and 1.0 diverges
        generate_Z: str = "canonic",
        add_iter_theta: int = 20,
        device: str = None,
    ):
        self.name = (
            "USNA"
            + (f" ν={alpha}")
            + (f" γ={gamma}")
            + (" Z~" + generate_Z if generate_Z != "normal" else "")
        )
        self.nu = alpha
        self.c_nu = c_alpha
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.add_iter_theta = add_iter_theta

        # If Z is a random vector of canonic base, we can compute faster P and Q
        self.generate_Z = generate_Z
        if generate_Z == "normal":
            self.update_hessian = self.update_hessian_normal
        elif generate_Z == "canonic" or generate_Z == "canonic deterministic":
            self.update_hessian = self.update_hessian_canonic
        else:
            raise ValueError(
                "Invalid value for Z. Choose 'normal', 'canonic' or 'canonic deterministic'."
            )

        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.size(0)
        self.hessian_inv = torch.eye(self.theta_dim, device=self.device)
        if self.generate_Z == "canonic deterministic":
            self.k = 0

    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        theta: torch.Tensor,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(data, theta)

        self.update_hessian(hessian)

        learning_rate_theta = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        theta += -learning_rate_theta * self.hessian_inv @ grad

    def update_hessian_normal(self, hessian: torch.Tensor):
        """
        Update the hessian estimate with a normal random vector
        """
        Z = torch.randn(self.theta_dim, device=self.device)
        P = self.hessian_inv @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_theta) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)
        if torch.dot(Q, Q) * torch.dot(Z, Z) <= beta**2:
            product = torch.outer(P, Q)
            self.hessian_inv += -learning_rate_hessian * (
                product + product.t() - 2 * torch.eye(self.theta_dim)
            )

    def update_hessian_canonic(self, hessian: torch.Tensor):
        """
        Update the hessian estimate with a canonic random vector
        """
        if self.generate_Z == "canonic":
            z = random.randint(0, self.theta_dim - 1)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k += 1
            self.k = self.k % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv[:, z]
        Q = hessian[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_theta) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)
        if torch.dot(Q, Q) * self.theta_dim <= beta**2:
            product = self.theta_dim * torch.outer(P, Q)
            self.hessian_inv += -learning_rate_hessian * (
                product + product.t() - 2 * torch.eye(self.theta_dim)
            )
