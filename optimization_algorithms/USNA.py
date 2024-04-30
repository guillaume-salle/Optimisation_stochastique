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
        Z: str = "canonic",
        initial_iteration: int = 20,
    ):
        self.name = f"USNA mu={mu} gamma={gamma}"
        self.mu = mu
        self.c_mu = c_mu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.initial_iteration = initial_iteration

        # Strategy selection based on Z
        if Z == "normal":
            self.step = self.step_normal
        elif Z == "canonic":
            self.step = self.step_canonic
        else:
            raise ValueError("Invalid value for Z. Choose 'normal' or 'canonic'.")

    def reset(self, theta_dim: int):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iteration = self.initial_iteration
        self.theta_dim = theta_dim  # To generate Z
        # TODO
        # Multiplier inverse de la hessienne par 100, ou initialiser iteration a 50
        self.hessian_inv = np.eye(theta_dim)

    def step_normal(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iteration += 1
        grad, hessian = g.grad_and_hessian(X, Y, theta)

        Z = np.random.randn(self.theta_dim)
        P = self.hessian_inv @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * self.iteration ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = P @ Q
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )

        learning_rate_theta = self.c_mu * self.iteration ** (-self.mu)
        theta += -learning_rate_theta * self.hessian_inv @ grad

    def step_canonic(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, g: BaseObjectiveFunction
    ):
        """
        Perform one optimization step
        """
        self.iteration += 1
        grad, hessian = g.grad_and_hessian(X, Y, theta)

        z = np.random.randint(0, self.theta_dim)
        P = self.hessian_inv[:, z]
        Q = hessian[:, z]
        learning_rate_hessian = self.c_gamma * self.iteration ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) <= self.theta_dim * beta**2:
            product = self.theta_dim * P @ Q
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )

        learning_rate_theta = self.c_mu * self.iteration ** (-self.mu)
        theta += -learning_rate_theta * self.hessian_inv @ grad
