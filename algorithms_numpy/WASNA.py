import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class WASNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 0.75,
        c_nu: float = 1.0,
        tau_theta: float = 2.0,
        add_iter_theta: int = 20,
        lambda_: int = 10,  # Weight more the initial identity matrix by lambda * d
        compute_hessian_theta_avg: bool = True,  # Where to compute the hessian
    ):
        self.name = (
            ("WASNA" if tau_theta != 0.0 else "SNA*")
            + (f" α={nu}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
            + (" NAT" if not compute_hessian_theta_avg else "")
        )
        self.alpha = nu
        self.c_alpha = c_nu
        self.tau_theta = tau_theta
        self.add_iter_theta = add_iter_theta
        self.lambda_ = lambda_
        self.compute_hessian_theta_avg = compute_hessian_theta_avg

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
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        if self.compute_hessian_theta_avg:  # cf article
            grad = g.grad(data, self.theta_not_avg)
            hessian = g.hessian(data, theta)
        else:
            grad, hessian = g.grad_and_hessian(data, self.theta_not_avg)

        # Update the hessian estimate
        self.hessian_bar += hessian
        hessian_inv = np.linalg.inv(self.hessian_bar / (self.iter + self.lambda_ * self.theta_dim))

        # Update the theta estimate
        learning_rate = self.c_alpha * (self.iter + self.add_iter_theta) ** (-self.alpha)
        self.theta_not_avg += -learning_rate * hessian_inv @ grad
        weight_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta
