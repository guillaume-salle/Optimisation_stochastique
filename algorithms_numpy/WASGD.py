import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class WASGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(
        self,
        nu: float = 0.75,
        c_nu: float = 5.0,
        tau: float = 2.0,
        add_iter_lr: int = 200,
    ):
        self.name = (
            ("WASGD" if tau != 0.0 else "ASGD")
            + (f" ν={nu}")
            + (f" τ={tau}" if tau != 0.0 and tau != 2.0 else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.tau = tau
        self.add_iter_lr = add_iter_lr  # Dont start at 0 to avoid large learning rates at the beginning

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the optimizer state
        """
        self.iter = 0
        self.theta_not_averaged = np.copy(initial_theta)
        self.sum_weights = 0

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
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
