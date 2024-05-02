import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        mu: float,
        c_mu: float = 1.0,
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = "SNA-Riccati" + rf" \mu={mu}"
        self.mu = mu
        self.c_mu = c_mu
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        # Weight more the initial identity matrix
        self.hessian_bar_inv = (
            1 / (self.lambda_ * self.theta_dim) * np.eye(self.theta_dim)
        )

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
        grad, phi = g.grad_and_riccati(X, Y, theta_estimate)
        product = self.hessian_bar_inv @ phi
        self.hessian_bar_inv += -np.outer(product, product) / (1 + np.dot(phi, product))
        learning_rate = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        theta_estimate += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )
