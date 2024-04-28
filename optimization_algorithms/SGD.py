import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(self, mu: float, c_mu: float):
        self.name = "SGD"
        self.mu = mu
        self.c_mu = c_mu
        self.iteration = 0

    def reset(self, theta_dim: int):
        """
        Reset the optimizer state
        """
        self.iteration = 0

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
        learning_rate = self.c_mu * self.iteration ** (-self.mu)
        theta += -learning_rate * grad
