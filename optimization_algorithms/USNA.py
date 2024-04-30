import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        mu: float,
        c_mu: float = 1.0,
        gamma: float = 1.0,
        c_gamma: float = 1.0,
        generate_Z: str = "canonic",
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix by lambda_ * d
    ):
        self.name = f"USNA mu={mu} gamma={gamma}"
        self.mu = mu
        self.c_mu = c_mu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

        # If Z is a random vector of canonic base, we can compute faster P and Q
        if generate_Z == "normal":
            self.step = self.step_normal
        elif generate_Z == "canonic":
            self.step = self.step_canonic
        else:
            raise ValueError("Invalid value for Z. Choose 'normal' or 'canonic'.")

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]  # needed by step to generate Z
        self.hessian_inv = self.lambda_ * self.theta_dim * np.eye(self.theta_dim)
        self.c_mu_lambda = self.c_mu / (
            self.lambda_ * self.theta_dim
        )  # Compensate lambda_ in constant

    def step_normal(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(X, Y, theta)

        Z = np.random.randn(self.theta_dim)
        P = self.hessian_inv @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = P @ Q
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )

        learning_rate_theta = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        theta += -learning_rate_theta * self.hessian_inv @ grad

    def step_canonic(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(X, Y, theta)

        z = np.random.randint(0, self.theta_dim)
        P = self.hessian_inv[:, z]
        Q = hessian[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) <= self.theta_dim * beta**2:
            product = self.theta_dim * P @ Q
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )

        learning_rate_theta = self.c_mu * (self.iter + self.add_iter_lr) ** (-self.mu)
        theta += -learning_rate_theta * self.hessian_inv @ grad
