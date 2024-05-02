import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(
        self, mu: float, c_mu: float = 1.0, thau: float = 1.0, add_iter_lr: int = 20
    ):
        self.name = (
            ("WASGD" if thau != 0.0 else "ASGD")
            + (rf" \mu={mu}" if mu != 1.0 else "")
            + (rf" \thau={thau}" if thau != 1.0 and thau != 0.0 else "")
        )
        self.mu = mu
        self.c_mu = c_mu
        self.thau = thau
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
        learning_rate = self.c_mu * ((self.iter + self.add_iter_lr) ** (-self.mu))
        self.theta_not_averaged += -learning_rate * grad

        weight = np.log(self.iter + 1) ** self.thau
        self.sum_weights += weight
        theta += (self.theta_not_averaged - theta) * weight / self.sum_weights
