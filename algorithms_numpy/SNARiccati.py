import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class SNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 1.0,
        c_nu: float = 1.0,
        add_iter_theta: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = "SNARiccati" + f" α={nu}"
        self.alpha = nu
        self.c_alpha = c_nu
        self.add_iter_theta = add_iter_theta
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
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta_estimate: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, phi = g.grad_and_riccati(data, theta_estimate, self.iter)
        product = self.hessian_bar_inv @ phi
        denominator = 1 + np.dot(phi, product)
        self.hessian_bar_inv += -np.outer(product, product) / denominator
        learning_rate = self.c_alpha * (self.iter + self.add_iter_theta) ** (-self.alpha)
        theta_estimate += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )
