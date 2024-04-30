import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class RiccatiWASNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        mu: float,
        c_mu: float = 1.0,
        add_iter_lr: int = 20,
        _lambda: float = 100.0,  # Weight more the initial identity matrix
    ):
        self.name = f"SNA mu={mu}"
        self.mu = mu
        self.c_mu = c_mu
        self.add_iter_lr = add_iter_lr
        self._lambda = _lambda

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        theta_dim = initial_theta.shape[0]
        self.hessian_inv = np.eye(theta_dim)

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
        grad, phi = g.grad_and_hessian(X, Y, theta_estimate)
        product = self.hessian_inv @ phi
        self.hessian_inv += -np.outer(product, product) / (1 + np.dot(phi, product))
        learning_rate = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        theta_estimate += -learning_rate * self.hessian_inv @ grad
