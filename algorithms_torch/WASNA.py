import torch
from typing import Tuple
import math

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class WASNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        alpha: float = 0.66,
        c_alpha: float = 1.0,
        tau_theta: float = 2.0,
        add_iter_theta: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix
        device: str = None,
    ):
        self.name = (
            ("WASNA" if tau_theta != 0.0 else "SNA*")
            + (f" ν={alpha}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
        )
        self.nu = alpha
        self.c_nu = c_alpha
        self.tau_theta = tau_theta
        self.add_iter_theta = add_iter_theta
        self.lambda_ = lambda_
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.size(0)
        self.theta_not_avg = initial_theta.clone().to(self.device)
        self.sum_weights_theta = 0
        # Weight more the initial identity matrix
        self.hessian_bar = self.lambda_ * self.theta_dim * torch.eye(self.theta_dim)

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

        # Update the hessian estimate
        self.hessian_bar += hessian
        hessian_inv = torch.inverse(self.hessian_bar / (self.iter + self.lambda_ * self.theta_dim))

        # Update the theta estimate
        learning_rate = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        self.theta_not_avg += -learning_rate * hessian_inv @ grad
        weight_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta
