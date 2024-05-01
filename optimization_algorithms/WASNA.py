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
        thau_theta: float = 1.0,
        thau_hessian: float = 1.0,
        add_iter_lr: int = 20,
        lambda_: float = 1.0,  # Weight more the initial identity matrix
    ):
        self.name = f"SNA mu={mu}"
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
        theta_dim = initial_theta.shape[0]
        self.theta_not_averaged = np.copy(initial_theta)
        self.sum_avg_coeff_theta = 0
        self.hessian_bar = self.lambda_ * np.eye(theta_dim)
        self.hessian_inv_not_averaged = np.eye(theta_dim)
        self.hessian_inv = np.eye(theta_dim)
        self.sum_avg_coeff_hessian = 0

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
        grad, hessian = g.grad_and_hessian(X, Y, theta)

        # Update the hessian estimate
        self.hessian_bar += hessian
        try:
            self.hessian_inv_not_averaged = np.linalg.inv(
                self.hessian_bar / (self.iter + self.lambda_)
            )
        except np.linalg.LinAlgError:
            print("Hessian is not invertible")
        avg_coeff_hessian = np.log(self.iter + 1) ** self.thau_hessian
        self.sum_avg_coeff_hessian += avg_coeff_hessian
        self.hessian_inv += (
            (self.hessian_inv_not_averaged - self.hessian_inv)
            * avg_coeff_hessian
            / self.sum_avg_coeff_hessian
        )

        # Update the theta estimate
        learning_rate = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        self.theta_not_averaged += -learning_rate * self.hessian_inv @ grad
        avg_coeff_theta = np.log(self.iter + 1) ** self.thau_theta
        self.sum_avg_coeff_theta += avg_coeff_theta
        theta += (
            (self.theta_not_averaged - theta)
            * avg_coeff_theta
            / self.sum_avg_coeff_theta
        )
