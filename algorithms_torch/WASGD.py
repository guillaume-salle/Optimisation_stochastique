import torch
import math
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class WASGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(
        self,
        nu: float,
        c_mu: float = 1.0,
        tau: float = 2.0,
        add_iter_lr: int = 20,
        device: str = None,
    ):
        self.name = (
            ("WASGD" if tau != 0.0 else "ASGD")
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" τ={tau}" if tau != 2.0 and tau != 0.0 else "")
        )
        self.nu = nu
        self.c_nu = c_mu
        self.tau = tau
        self.add_iter_lr = add_iter_lr  # Dont start at 0 to avoid large learning rates at the beginning
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset(self, initial_theta: torch.Tensor):
        """
        Reset the optimizer state
        """
        self.iter = 0
        self.theta_not_averaged = initial_theta.clone().to(self.device)
        self.sum_weights = 0

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
        grad = g.grad(data, self.theta_not_averaged)
        learning_rate = self.c_nu * ((self.iter + self.add_iter_lr) ** (-self.nu))
        self.theta_not_averaged += -learning_rate * grad

        weight = math.log(self.iter + 1) ** self.tau
        self.sum_weights += weight
        theta += (self.theta_not_averaged - theta) * weight / self.sum_weights
