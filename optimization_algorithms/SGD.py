import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(self, mu: float, c_mu: float = 1.0, initial_iteration: int = 20):
        self.name = f"SGD mu={mu}"
        self.mu = mu
        self.c_mu = c_mu
        self.initial_iteration = initial_iteration  # Dont start at 0 to avoid large learning rates at the beginning

    def reset(self, theta_dim: int):
        """
        Reset the optimizer state
        """
        self.iteration = self.initial_iteration

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
        self.iteration += 1
        grad = g.grad(X, Y, theta)
        learning_rate = self.c_mu * (self.iteration ** (-self.mu))
        theta += -learning_rate * grad
