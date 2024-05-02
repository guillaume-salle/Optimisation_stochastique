import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class SNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float,
        c_nu: float = 1.0,
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = "SNA-Riccati" + f" ν={nu}"
        self.nu = nu
        self.c_nu = c_nu
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        # Weight more the initial identity matrix
        self.hessian_bar_inv = np.eye(self.theta_dim) / (self.lambda_ * self.theta_dim)

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
        grad, phi = g.grad_and_riccati(X, Y, theta_estimate, self.iter)
        product = self.hessian_bar_inv @ phi
        denominator = 1 + np.dot(phi, product)
        if np.abs(denominator) < 1e-8:
            print("Denominator too small, update skipped")
        else:
            self.hessian_bar_inv += -np.outer(product, product) / denominator
        learning_rate = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        theta_estimate += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )
