import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(
        self, nu: float, c_mu: float = 1.0, tau: float = 2.0, add_iter_lr: int = 20
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

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the optimizer state
        """
        self.iter = 0
        self.theta_not_averaged = np.copy(initial_theta)
        self.sum_weights = 0

    def step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(X, Y, self.theta_not_averaged)
        learning_rate = self.c_nu * ((self.iter + self.add_iter_lr) ** (-self.nu))
        self.theta_not_averaged += -learning_rate * grad

        weight = np.log(self.iter + 1) ** self.tau
        self.sum_weights += weight
        theta += (self.theta_not_averaged - theta) * weight / self.sum_weights
