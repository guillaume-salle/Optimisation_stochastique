import numpy as np
from typing import Any

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 1.0,  # Set to 1.0 in the article
        c_nu: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, and 1.0 diverges
        generate_Z: str = "normal",
        add_iter_lr: int = 20,
    ):
        self.name = (
            "USNA"
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" γ={gamma}")
            + (" Z~" + generate_Z if generate_Z != "normal" else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.add_iter_lr = add_iter_lr

        # If Z is a random vector of canonic base, we can compute faster P and Q
        self.generate_Z = generate_Z
        if generate_Z == "normal":
            self.update_hessian = self.update_hessian_normal
        elif generate_Z == "canonic" or generate_Z == "canonic deterministic":
            self.update_hessian = self.update_hessian_canonic
        else:
            raise ValueError(
                "Invalid value for Z. Choose 'normal', 'canonic' or 'canonic deterministic'."
            )

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        self.hessian_inv = np.eye(self.theta_dim)
        if self.generate_Z == "canonic deterministic":
            self.k = 0

    def step(self, data: Any, theta: np.ndarray, g: BaseObjectiveFunction):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, hessian = g.grad_and_hessian(data, theta)

        self.update_hessian(hessian)

        learning_rate_theta = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        theta += -learning_rate_theta * self.hessian_inv @ grad

    def update_hessian_normal(self, hessian: np.ndarray):
        """
        Update the hessian estimate with a normal random vector
        """
        Z = np.random.randn(self.theta_dim)
        P = self.hessian_inv @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(P, Q)
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )

    def update_hessian_canonic(self, hessian: np.ndarray):
        """
        Update the hessian estimate with a canonic random vector
        """
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k += 1
            self.k = self.k % self.theta_dim
        else:
            raise ValueError(
                "Invalid value for Z. Choose 'canonic' or 'canonic deterministic'."
            )
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv[:, z]
        Q = hessian[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * self.theta_dim <= beta**2:
            product = self.theta_dim * np.outer(P, Q)
            self.hessian_inv += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )
