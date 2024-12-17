import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class USNA_old(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        param: np.ndarray,
        objective_function: BaseObjectiveFunction,
        nu: float = 1.0,  # Set to 1.0 in the article
        c_nu: float = 1.0,  # Set to 1.0 in the article
        gamma: float = 0.75,  # Set to 0.75 in the article
        c_gamma: float = 0.1,  # Not specified in the article, and 1.0 diverges
        generate_Z: str = "canonic",
        add_iter_theta: int = 0,
        add_iter_hessian: int = 200,  # Works better
        sym: bool = True,  # Symmetric estimate of the hessian
        algo: str = "rapport",  # Version of USNA described in the rapport, by default
    ):
        self.name = (
            "USNA"
            + "old"
            + (f" α={nu}")
            + (f" γ={gamma}" if gamma != 0.75 else "")
            + (" " + algo if algo != "rapport" else "")
            + (" Z~" + generate_Z if generate_Z != "canonic" else "")
            + (" NS" if not sym else "")
        )
        self.alpha = nu
        self.c_alpha = c_nu
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.generate_Z = generate_Z
        self.add_iter_theta = add_iter_theta
        self.add_iter_hessian = add_iter_hessian
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

        self.reset(param)
        self.param = param
        self.objective_function = objective_function

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
    ):
        theta = self.param
        g = self.objective_function
        """
        Perform one optimization step
        """
        self.iter += 1

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(g, data, theta)

        # Update theta
        learning_rate_theta = self.c_alpha * (self.iter + self.add_iter_theta) ** (-self.alpha)
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

        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(Q, P)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))
                norm = np.linalg.norm(self.hessian_inv, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                beta_prime = 1 / (4 * learning_rate_hessian)
                if norm > beta_prime:
                    self.hessian_inv *= beta_prime / norm

            # Version with added term to ensure positive definiteness like in rapport, but right multiplication
            elif self.algo == "right":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.hessian_inv, Z
                    ) * np.outer(
                        Q, Q
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.hessian_inv, Z
                    ) * np.outer(
                        Q, Q
                    )
            else:
                raise ValueError(
                    "Invalid value for algo. Choose 'article', 'article v2' or 'right'."
                )

        return grad

    # Different function because if we know Z is a canonic vector,
    # some computations can a be simple column/coeff selection
    def update_hessian_canonic_right(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        Old version, the multiplication by Z Z^T is done on the RIGHT of the random hessian h_t
        """
        # Generate Z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.theta_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            # Update the index for the next iteration
            self.k = (self.k + 1) % self.theta_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # Compute vectors P and Q
        grad, Q = g.grad_and_hessian_column(data, theta, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later after checking <=beta
        P = self.hessian_inv[:, z]
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        # An alternative to bruno's added term would be the following condition (not studied yet):
        # update_condition =  np.dot(Q, Q) * self.theta_dim**2 * P[z] <= beta**2:
        update_condition = np.dot(Q, Q) * self.theta_dim**2 <= beta**2
        if update_condition:
            product = self.theta_dim * np.outer(Q, P)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    )
                else:
                    self.hessian_inv += -learning_rate_hessian * (product - np.eye(self.theta_dim))

                # Projection step:
                norm = np.linalg.norm(self.hessian_inv, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                if norm > 1 / (4 * learning_rate_hessian):
                    self.hessian_inv /= norm

            # version with added term to ensure positive definiteness
            elif self.algo == "right":
                if self.sym:
                    self.hessian_inv += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)
                else:
                    self.hessian_inv += -learning_rate_hessian * (
                        product - np.eye(self.theta_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)

        return grad

    def update_hessian_canonic_left(
        self,
        g: BaseObjectiveFunction,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
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
        grad, Q = g.grad_and_hessian_column(data, theta, z)
        # Z is supposed to be sqrt(theta_dim) * e_z, but will multiply later
        learning_rate_hessian = self.c_gamma * (self.iter + self.add_iter_hessian) ** (-self.gamma)
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * self.theta_dim**2 <= beta**2:
            product = self.theta_dim * Q.T @ self.hessian_inv

            # rapport version, with the added term and the left multiplication by ZZ^T
            if self.algo == "rapport":
                self.hessian_inv[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.hessian_inv[:, z] += -learning_rate_hessian * product
                self.hessian_inv += (1 + self.sym) * learning_rate_hessian * np.eye(self.theta_dim)
                self.hessian_inv[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.hessian_inv, Q
                )

            # Z Z^T replace Id, on top of rapport version
            elif self.algo == "proj":
                product[z] += -self.theta_dim
                self.hessian_inv[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.hessian_inv[:, z] += -learning_rate_hessian * product
                self.hessian_inv[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.hessian_inv, Q
                )

        return grad
