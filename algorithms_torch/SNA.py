import torch
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class SNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        alpha: float,
        c_alpha: float = 1.0,
        add_iter_theta: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = "SNA" + f" ν={alpha}"
        self.nu = alpha
        self.c_nu = c_alpha
        self.add_iter_theta = add_iter_theta
        self.lambda_ = lambda_

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.size(0)
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
        self.hessian_bar += hessian
        hessian_inv = torch.inverse(self.hessian_bar / (self.iter + self.lambda_ * self.theta_dim))
        learning_rate = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        theta += -learning_rate * hessian_inv @ grad
