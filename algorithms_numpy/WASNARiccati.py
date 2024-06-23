import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class WASNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 0.75,
        c_nu: float = 1.0,
        tau_theta: float = 2.0,
        add_iter_theta: int = 200,
        lambda_: float = 10.0,  # Weight more the initial identity matrix
        compute_hessian_theta_avg: bool = True,  # Where to compute the hessian
    ):
        self.class_name = "WASNARiccati"
        self.name = (
            ("WASNARiccati" if tau_theta != 0.0 else "SNARiccati")
            + (f" ν={nu}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
            + (" NAT" if not compute_hessian_theta_avg else "")
        )
        self.nu = nu
        self.c_nu = c_nu
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
        self.hessian_bar_inv = 1 / (self.lambda_ * self.theta_dim) * np.eye(self.theta_dim)

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
            phi = g.riccati(data, theta, self.iter)
        else:
            grad, phi = g.grad_and_riccati(data, self.theta_not_avg, self.iter)

        # Update the hessian estimate
        product = self.hessian_bar_inv @ phi
        denominator = 1 + np.dot(phi, product)
        self.hessian_bar_inv += -np.outer(product, product) / denominator

        # Update the not averaged theta
        learning_rate = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        self.theta_not_avg += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )

        # Update the averaged theta
        weigth_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weigth_theta
        theta += (self.theta_not_avg - theta) * weigth_theta / self.sum_weights_theta
