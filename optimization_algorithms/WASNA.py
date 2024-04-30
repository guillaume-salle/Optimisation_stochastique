import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        mu: float,
        c_mu: float = 1.0,
        add_iter_lr: int = 20,
        lambda_: float = 1.0,  # Weight more the initial identity matrix
    ):
        self.name = f"SNA mu={mu}"
        self.mu = mu
        self.c_mu = c_mu
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        theta_dim = initial_theta.shape[0]
        self.hessian_bar = self.lambda_ * np.eye(theta_dim)
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
        grad, hessian = g.grad_and_hessian(X, Y, theta_estimate)
        self.hessian_bar += hessian
        try:
            self.hessian_inv = np.linalg.inv(
                self.hessian_bar / (self.iter + self.lambda_)
            )
        except np.linalg.LinAlgError:
            print("Hessian is not invertible")
        learning_rate = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        theta_estimate += -learning_rate * self.hessian_inv @ grad
