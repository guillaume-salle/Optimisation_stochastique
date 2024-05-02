import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(self, mu: float, c_mu: float = 1.0, add_iter_lr: int = 20):
        self.name = "SGD" + rf" \mu={mu}"
        self.mu = mu
        self.c_mu = c_mu
        self.add_iter_lr = add_iter_lr  # Dont start at 0 to avoid large learning rates at the beginning

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the optimizer state
        """
        self.iter = 0

    def step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        theta_estimate: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(X, Y, theta_estimate)
        learning_rate = self.c_mu * ((self.iter + self.add_iter_lr) ** (-self.mu))
        theta_estimate += -learning_rate * grad
