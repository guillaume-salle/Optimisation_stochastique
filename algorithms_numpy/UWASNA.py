import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class UWASNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    # Heriter de la classe USNA
    def __init__(
        self,
        nu: float = 0.75,  # Do not use 1 for averaged algo
        c_nu: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, 1.0 diverges
        tau_theta: float = 2.0,  # Not specified in the article
        tau_hessian: float = 2.0,  # Not specified in the article
        generate_Z: str = "canonic",
        add_iter_theta: int = 200,
        add_iter_hessian: int = 200,
        compute_hessian_theta_avg: bool = True,  # Where to compute the hessian
        use_hessian_avg: bool = True,  # Use the averaged hessian
        sym: bool = True,  # Symmetric estimate of the hessian
        algo: str = "rapport",  # Version of UWASNA described in the rapport, by default
    ):
        self.name = (
            ("UWASNA" if (tau_theta != 0.0 or tau_hessian != 0.0) else "USNA")
            + (f" α={nu}")
            + (f" γ={gamma}")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
            + (f" τ_hessian={tau_hessian}" if tau_hessian != 2.0 and tau_theta != 0.0 else "")
            + (" " + algo if algo != "rapport" else "")
            + (" Z~" + generate_Z if generate_Z != "canonic" else "")
            + (" NAT" if not compute_hessian_theta_avg else "")
            + (" NAH" if not use_hessian_avg else "")
            + (" NS" if not sym else "")
        )
        self.alpha = nu
        self.c_alpha = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.generate_Z = generate_Z
        self.tau_theta = tau_theta
        self.tau_hessian = tau_hessian
        self.add_iter_theta = add_iter_theta
        self.add_iter_hessian = add_iter_hessian
        self.compute_hessian_theta_avg = compute_hessian_theta_avg
        self.use_hessian_avg = use_hessian_avg
        self.sym = sym
        self.algo = algo

        # Test different versions of the algorithm
        # 'article', 'article v2' for the revised version with frobenius ball projection
        # 'rapport' for the added term with left multiplication by ZZ^T, 'right' for right multiplication
        # 'proj' for left multiplication and ZZ^T au lieu de Id, not studied in theory yet
        versions = ["article", "article v2", "rapport", "right", "proj"]
        if algo not in versions:
            raise ValueError("Invalid value for algo. Choose " + ", ".join(versions) + ".")

        # In case we know Z is a random vector of canonic base, we can compute faster P and Q
        if generate_Z == "normal":
            self.update_hessian = self.update_hessian_normal
            if self.algo == "rapport" or self.algo == "proj":
                raise ValueError(
                    "Invalid value for algo. No point in multiplying on the left with a normal vector."
                )
        elif generate_Z == "canonic" or generate_Z == "canonic deterministic":
            # Different fonction if we multiply by Z Z^T on the left or on the right
            if self.algo == "rapport" or self.algo == "proj":
                self.update_hessian = self.update_hessian_canonic_left
            else:
                self.update_hessian = self.update_hessian_canonic_right
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

        # Update the not averaged hessian estimate
        # and get the gradient from intermediate computation
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

        learning_rate_theta = self.c_alpha * (self.iter + self.add_iter_theta) ** (-self.alpha)
        if self.use_hessian_avg:
            self.theta_not_avg += -learning_rate_theta * self.hessian_inv @ grad
        else:
            self.theta_not_avg += -learning_rate_theta * self.hessian_inv_not_avg @ grad
        weight_theta = math.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weight_theta
        theta += (self.theta_not_avg - theta) * weight_theta / self.sum_weights_theta

    # The rest is almost copied from USNA
    # TODO: factorize code

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
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(P, Q)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    )

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    )
                norm = np.linalg.norm(self.hessian_inv_not_avg, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                beta_prime = 1 / (4 * learning_rate_hessian)
                if norm > beta_prime:
                    self.hessian_inv_not_avg *= beta_prime / norm

            # Version with added term to ensure positive definiteness like in rapport, but right multiplication
            elif self.algo == "right":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.hessian_inv_not_avg, Z
                    ) * np.outer(
                        Q, Q
                    )
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.hessian_inv_not_avg, Z
                    ) * np.outer(
                        Q, Q
                    )
            else:
                raise ValueError(
                    "Invalid value for algo. Choose 'article', 'article v2' or 'right'."
                )

        return grad

    def update_hessian_canonic_right(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ):
        """
        Update the not averaged hessian estimate with a canonic base random vector, also returns grad
        Old version, the multiplication by Z Z^T is done on the RIGHT of the random hessian h_t
        """
        # Generate Z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # Compute vectors P and Q
        if self.compute_hessian_theta_avg:  # cf article
            grad = g.grad(data, self.theta_not_avg)
            Q = g.hessian_column(data, theta, z)
        else:
            grad, Q = g.grad_and_hessian_column(data, self.theta_not_avg, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        P = self.hessian_inv_not_avg[:, z]  # Use the non averaged hessian to compute P
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * np.outer(P, Q)  # Multiply by the dimension

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    )

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    )

                # Projection step:
                norm = np.linalg.norm(self.hessian_inv_not_avg, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                if norm > 1 / (4 * learning_rate_hessian):
                    self.hessian_inv_not_avg /= norm

            # version with added term to ensure positive definiteness
            elif self.algo == "right":
                if self.sym:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)
                else:
                    self.hessian_inv_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)

        return grad

    def update_hessian_canonic_left(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ):
        """
        Update the not averaged hessian estimate with a canonic base random vector, also returns grad
        New version, the multiplication by Z Z^T is done on the LEFT of the random hessian h_t
        Also new projected version, close to multiply on left so i put it here, not studied yet
        """
        # Generate Z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # TODO: we want a line here, not a column. Do a grad_and_hessian_iine function
        # It is the same if random hessians are symmetric, which is the case in our simulations.
        if self.compute_hessian_theta_avg:  # cf article
            grad = g.grad(data, self.theta_not_avg)
            Q = g.hessian_column(data, theta, z)
        else:
            grad, Q = g.grad_and_hessian_column(data, self.theta_not_avg, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * Q.T @ self.hessian_inv_not_avg

            # rapport version, with the added term and the left multiplication by ZZ^T
            if self.algo == "rapport":
                self.hessian_inv_not_avg[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.hessian_inv_not_avg[:, z] += -learning_rate_hessian * product
                self.hessian_inv_not_avg += (
                    (1 + self.sym) * learning_rate_hessian * np.eye(self.theta_dim)
                )
                self.hessian_inv_not_avg[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.hessian_inv_not_avg, Q
                )

            # Z Z^T replace Id, on top of rapport version
            elif self.algo == "proj":
                product[z] += -self.theta_dim
                self.hessian_inv_not_avg[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.hessian_inv_not_avg[:, z] += -learning_rate_hessian * product
                self.hessian_inv_not_avg[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.hessian_inv_not_avg, Q
                )

        return grad
