import torch
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(self, nu: float, c_nu: float = 1.0, add_iter_theta: int = 20):
        self.name = "SGD" + f" ν={nu}"
        self.nu = nu
        self.c_nu = c_nu
        self.add_iter_theta = (
            add_iter_theta  # Dont start at 0 to avoid large learning rates at the beginning
        )

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the optimizer state
        """
        self.iter = 0

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
        grad = g.grad(data, theta)
        learning_rate = self.c_nu * ((self.iter + self.add_iter_theta) ** (-self.nu))
        theta += -learning_rate * grad
