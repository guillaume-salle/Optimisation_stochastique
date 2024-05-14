import numpy as np
from typing import Tuple

from algorithms_torch_streaming import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


class SNA(BaseOptimizer):
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
        self.name = "SNA" + f" ν={nu}"
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
        self.hessian_bar = self.lambda_ * self.theta_dim * np.eye(self.theta_dim)

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(data, theta)
        self.hessian_bar += hessian
        hessian_inv = np.linalg.inv(
            self.hessian_bar / (self.iter + self.lambda_ * self.theta_dim)
        )
        learning_rate = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        theta += -learning_rate * hessian_inv @ grad
