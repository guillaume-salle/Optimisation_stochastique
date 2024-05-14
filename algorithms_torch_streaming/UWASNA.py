import torch
import math
import random
from typing import Tuple

from algorithms_torch_streaming import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class UWASNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 0.75,  # Do not take 1 for averaged algorithms
        c_nu: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, 1.0 diverges
        tau_theta: float = 2.0,  # Not specified in the article
        tau_hessian: float = 2.0,  # Not specified in the article
        generate_Z: str = "canonic",
        add_iter_lr: int = 50,
        device: str = None,
    ):
        self.name = (
            ("UWASNA" if tau_theta != 0.0 or tau_hessian != 0.0 else "USNA")
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" γ={gamma}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
            + (
                f" τ_hessian={tau_hessian}"
                if tau_hessian != 2.0 and tau_theta != 0.0
                else ""
            )
            + (" Z~" + generate_Z if generate_Z != "normal" else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.tau_theta = tau_theta
        self.tau_hessian = tau_hessian
        self.add_iter_lr = add_iter_lr

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
        self.theta_dim = initial_theta.shape[0]
        self.theta_not_avg = initial_theta.detach().clone().to(self.device)
        self.sum_weights_theta = 0
        self.hessian_inv_not_avg = torch.eye(self.theta_dim, device=self.device)
        self.hessian_inv = torch.eye(self.theta_dim, device=self.device)
        self.sum_weights_hessian = 0
        if self.generate_Z == "canonic deterministic":
            self.k = 0

    def update_hessian_normal(self, hessian: torch.Tensor):
        """
        Update the hessian estimate with a normal random vector
        """
        Z = torch.randn(self.theta_dim, device=self.device)
        # Use the non averaged hessian to compute P
        P = self.hessian_inv_not_avg @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if torch.dot(Q, Q) * torch.dot(Z, Z) <= beta**2:
            product = torch.outer(P, Q)
            self.hessian_inv_not_avg += -learning_rate_hessian * (
                product + product.t() - 2 * torch.eye(self.theta_dim)
            )

    def update_hessian_canonic(self, hessian: torch.Tensor):
        """
        Update the hessian estimate with a canonic base random vector
        """
        if self.generate_Z == "canonic":
            z = random.randint(0, self.theta_dim - 1)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k += 1
            self.k = self.k % self.theta_dim
        else:
            raise ValueError(
                "Invalid value for Z. Choose 'canonic' or 'canonic deterministic'."
            )

        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv_not_avg[:, z]  # Use the non averaged hessian to compute P
        Q = hessian[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if torch.dot(Q, Q) * self.theta_dim <= beta**2:
            product = self.theta_dim * torch.outer(P, Q)  # Multiply by the dimension
            self.hessian_inv_not_avg += -learning_rate_hessian * (
                product + product.t() - 2 * torch.eye(self.theta_dim)
            )

    def step(
        self, data: Tuple | torch.Tensor, theta: torch.Tensor, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(data, self.theta_not_avg)  # gradient in the NOT averaged theta
        hessian = g.hessian(data, theta)  # hessian in the averaged theta

        self.update_hessian(hessian)

        # Update the averaged hessian
        weight_hessian = math.log(self.iter + 1) ** self.tau_hessian
        self.sum_weights_hessian += weight_hessian
        self.hessian_inv += (
            (self.hessian_inv_not_avg - self.hessian_inv)
            * weight_hessian
            / self.sum_weights_hessian
        )

        # Update the theta estimate with the averaged hessian
        learning_rate_theta = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        self.theta_not_avg += -learning_rate_theta * self.hessian_inv @ grad
        weight_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta
