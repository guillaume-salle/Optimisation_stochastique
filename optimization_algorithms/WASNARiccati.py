import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        mu: float,
        c_mu: float = 1.0,
        thau_theta: float = 1.0,
        thau_hessian: float = 1.0,
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix
    ):
        self.name = (
            "WASNA-Riccati"
            if thau_theta != 0.0 or thau_hessian != 0.0
            else "SNA-Riccati"
            + (rf" \mu={mu}" if mu != 1.0 else "")
            + (
                rf" \thau_theta={thau_theta}"
                if thau_theta != 1.0 and thau_theta != 0.0
                else ""
            )
            + (
                rf" \thau_hessian={thau_hessian}"
                if thau_hessian != 1.0 and thau_theta != 0.0
                else ""
            )
        )
        self.mu = mu
        self.c_mu = c_mu
        self.thau_theta = thau_theta
        self.thau_hessian = thau_hessian
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
        self.hessian_bar_inv = (
            1 / (self.lambda_ * self.theta_dim) * np.eye(self.theta_dim)
        )
        self.hessian_inv = np.eye(self.theta_dim)
        self.sum_weights_hessian = 0

    def step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, phi = g.grad_and_riccati(X, Y, theta)

        # Update the hessian estimate
        product = self.hessian_bar_inv @ phi
        self.hessian_bar_inv += -np.outer(product, product) / (1 + np.dot(phi, product))
        weight_hessian = np.log(self.iter + 1) ** self.thau_hessian
        self.sum_weights_hessian += weight_hessian
        self.hessian_inv += (
            (
                (self.iter + self.lambda_ * self.theta_dim) * self.hessian_bar_inv
                - self.hessian_inv
            )
            * weight_hessian
            / self.sum_weights_hessian
        )

        # Update the theta estimate
        learning_rate = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        self.theta_not_avg += -learning_rate * self.hessian_inv @ grad
        weigth_theta = np.log(self.iter + 1) ** self.thau_theta
        self.sum_weights_theta += weigth_theta
        theta += (self.theta_not_avg - theta) * weigth_theta / self.sum_weights_theta