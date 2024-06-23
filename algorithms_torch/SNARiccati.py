import torch
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class SNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float,
        c_nu: float = 1.0,
        add_iter_theta: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = "SNARiccati" + f" ν={nu}"
        self.nu = nu
        self.c_nu = c_nu
        self.add_iter_theta = add_iter_theta
        self.lambda_ = lambda_

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        # Weight more the initial identity matrix
        self.hessian_bar_inv = torch.eye(self.theta_dim) / (self.lambda_ * self.theta_dim)

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
        grad, phi = g.grad_and_riccati(data, theta, self.iter)
        product = self.hessian_bar_inv @ phi
        denominator = 1 + torch.dot(phi, product)
        self.hessian_bar_inv += -torch.outer(product, product) / denominator
        learning_rate = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        theta += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )
