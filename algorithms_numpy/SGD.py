import numpy as np
from typing import Tuple

from algorithms_torch import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_alpha * iteration^(-alpha)
    """

    def __init__(self, nu: float, c_nu: float = 1.0, add_iter_theta: int = 20):
        self.name = "SGD" + f" α={nu}"
        self.alpha = nu
        self.c_alpha = c_nu
        self.add_iter_theta = (
            add_iter_theta  # Dont start at 0 to avoid large learning rates at the beginning
        )

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the optimizer state
        """
        self.iter = 0

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta_estimate: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(data, theta_estimate)
        learning_rate = self.c_alpha * ((self.iter + self.add_iter_theta) ** (-self.alpha))
        theta_estimate += -learning_rate * grad
