import numpy as np
from typing import Any

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 0.66,
        c_nu: float = 1.0,
        tau_theta: float = 2.0,
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix
    ):
        self.name = (
            ("WASNA" if tau_theta != 0.0 else "SNA*")
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.tau_theta = tau_theta
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        self.theta_not_avg = np.copy(initial_theta)
        self.sum_weights_theta = 0
        # Weight more the initial identity matrix
        self.hessian_bar = self.lambda_ * self.theta_dim * np.eye(self.theta_dim)

    def step(
        self,
        data: Any,
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(data, theta)

        # Update the hessian estimate
        self.hessian_bar += hessian
        try:
            hessian_inv = np.linalg.inv(
                self.hessian_bar / (self.iter + self.lambda_ * self.theta_dim)
            )
        except np.linalg.LinAlgError:
            # Hessian is not invertible
            hessian_inv = np.eye(self.theta_dim)

        # Update the theta estimate
        learning_rate = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        self.theta_not_avg += -learning_rate * hessian_inv @ grad
        weight_theta = np.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta
