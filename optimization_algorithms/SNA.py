import numpy as np
from typing import Callable

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(self, mu: float, c_mu: float):
        self.name = "SNA"
        self.mu = mu
        self.c_mu = c_mu
        self.iteration = 0

    def reset(self, theta_dim: int):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iteration = 0
        self.hessian = np.eye(theta_dim)
        self.hessian_inv = np.eye(theta_dim)

    def step(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iteration += 1
        grad, hessian = g.grad_and_hessian(X, Y, theta)
        self.hessian += (hessian - self.hessian) / (self.iteration + 1)
        try:
            self.hessian_inv = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print("Hessian is not invertible")
        learning_rate = self.c_mu * self.iteration ** (-self.mu)
        theta += -learning_rate * self.hessian_inv @ grad
