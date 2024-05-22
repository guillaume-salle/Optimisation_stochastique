import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class UWASNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float = 0.75,  # Do not use 1 for averaged algo
        c_nu: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, 1.0 diverges
        tau_theta: float = 2.0,  # Not specified in the article
        tau_hessian: float = 2.0,  # Not specified in the article
        generate_Z: str = "canonic",
        add_iter_lr: int = 200,
        compute_hessian_theta_avg: bool = True,  # Where to compute the hessian
        use_hessian_avg: bool = True,  # Use the averaged hessian
    ):
        self.name = (
            ("UWASNA" if (tau_theta != 0.0 or tau_hessian != 0.0) else "USNA")
            + (f" ν={nu}")
            + (f" γ={gamma}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
            + (
                f" τ_hessian={tau_hessian}"
                if tau_hessian != 2.0 and tau_theta != 0.0
                else ""
            )
            + (" Z~" + generate_Z if generate_Z != "canonic" else "")
            + (" NAT" if not compute_hessian_theta_avg else "")
            + (" NAH" if not use_hessian_avg else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.tau_theta = tau_theta
        self.tau_hessian = tau_hessian
        self.add_iter_lr = add_iter_lr
        self.compute_hessian_theta_avg = compute_hessian_theta_avg
        self.use_hessian_avg = use_hessian_avg

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
        d = initial_theta.shape[0]
        self.theta_dim = d
        self.theta_not_avg = np.copy(initial_theta)
        self.sum_weights_theta = 0
        self.hessian_inv_not_avg = np.eye(d)
        if self.use_hessian_avg:
            self.hessian_inv = np.eye(d)
            self.sum_weights_hessian = 0
        if self.generate_Z == "canonic deterministic":
            self.k = 0

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
        grad = g.grad(data, self.theta_not_avg)  # gradient in the NOT averaged theta

        # Update the not averaged hessian
        grad = self.update_hessian(g, data, theta)

        # Update the averaged hessian
        if self.use_hessian_avg:
            weight_hessian = np.log(self.iter + 1) ** self.tau_hessian
            self.sum_weights_hessian += weight_hessian
            self.hessian_inv += (
                (self.hessian_inv_not_avg - self.hessian_inv)
                * weight_hessian
                / self.sum_weights_hessian
            )

        learning_rate_theta = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        if self.use_hessian_avg:
            self.theta_not_avg += -learning_rate_theta * self.hessian_inv @ grad
        else:
            self.theta_not_avg += -learning_rate_theta * self.hessian_inv_not_avg @ grad
        weight_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta

    def update_hessian_normal(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the not averaged hessian estimate with a normal random vector, also returns grad
        """
        if self.compute_hessian_theta_avg:  # cf article
            grad = g.grad(data, self.theta_not_avg)
            hessian = g.hessian(data, theta)
        else:
            grad, hessian = g.grad_and_hessian(data, self.theta_not_avg)
        Z = np.random.standard_normal(self.theta_dim)
        P = self.hessian_inv_not_avg @ Z  # Use the non averaged hessian to compute P
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(P, Q)
            self.hessian_inv_not_avg += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )
        return grad

    def update_hessian_canonic(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ):
        """
        Update the not averaged hessian estimate with a canonic base random vector, also returns grad
        """
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError(
                "Invalid value for Z. Choose 'canonic' or 'canonic deterministic'."
            )

        if self.compute_hessian_theta_avg:  # cf article
            grad = g.grad(data, self.theta_not_avg)
            Q = g.hessian_column(data, theta, z)
        else:
            grad, Q = g.grad_and_hessian_column(data, self.theta_not_avg, z)

        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv_not_avg[:, z]  # Use the non averaged hessian to compute P
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_lr) ** (
            -self.gamma
        )
        beta = 1 / (2 * learning_rate_hessian)
        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * np.outer(P, Q)  # Multiply by the dimension
            self.hessian_inv_not_avg += -learning_rate_hessian * (
                product + product.transpose() - 2 * np.eye(self.theta_dim)
            )
        return grad
