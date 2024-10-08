import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


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
        generate_Z: str = "canonic",
        add_iter_theta: int = 20,
        add_iter_hessian: int = 200,  # Works better
        sym: bool = True,  # Symmetric estimate of the hessian
        algo: str = "article",
    ):
        self.name = (
            "USNA"
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" γ={gamma}" if gamma != 0.75 else "")
            + (" Z~" + generate_Z if generate_Z != "canonic" else "")
            + (" NS" if not sym else "")
            + (" " + algo if algo != "article" else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.add_iter_theta = add_iter_theta
        self.add_iter_hessian = add_iter_hessian
        self.sym = sym

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

        # TODO DELETE
        self.algo = algo
        if algo not in ["article", "bruno", "guillaume"]:
            raise ValueError("Invalid value for algo. Choose 'article', 'bruno' or 'guillaume'.")
        if algo == "guillaume":
            self.update_hessian = self.update_hessian_guillaume

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        self.hessian_inv = np.eye(self.theta_dim)
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

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(g, data, theta)

        # Update theta
        learning_rate_theta = self.c_nu * (self.iter + self.add_iter_theta) ** (-self.nu)
        theta += -learning_rate_theta * self.hessian_inv @ grad

    def update_hessian_normal(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the hessian estimate with a normal random vector, also returns grad
        """
        grad, hessian = g.grad_and_hessian(data, theta)
        Z = np.random.standard_normal(self.theta_dim)
        P = self.hessian_inv @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)
        # TODO: Check if the condition is correct
        # if np.dot(Q, Q) * np.dot(Z, P) <= beta**2:
        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(Q, P)
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))

            elif self.algo == "bruno":
                self.hessian_inv += -learning_rate_hessian * (
                    product + product.transpose() - 2 * np.eye(self.theta_dim)
                ) + learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Z, self.hessian_inv, Z
                ) * np.outer(
                    Q, Q
                )
        return grad

    # TODO: Factorize the code

    def update_hessian_canonic(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        """
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")
        grad, Q = g.grad_and_hessian_column(data, theta, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        # TODO: Check if the condition is correct
        # if np.dot(Q, Q) * self.theta_dim**2 * P[z] <= beta**2:
        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * np.outer(Q, P)
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))
            elif self.algo == "bruno":
                self.hessian_inv += -learning_rate_hessian * (
                    product + product.transpose() - 2 * np.eye(self.theta_dim)
                ) + learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", P, self.hessian_inv, P
                ) * np.outer(
                    Q, Q
                )

        return grad

    def update_hessian_guillaume(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        """
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")
        grad, Q = g.grad_and_hessian_column(data, theta, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        # TODO: Check if the condition is correct
        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * Q.T @ self.hessian_inv
            product[z] -= self.theta_dim
            self.hessian_inv[z, :] += -learning_rate_hessian * product
            if self.sym:
                self.hessian_inv[:, z] += -learning_rate_hessian * product

        return grad
