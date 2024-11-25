import numpy as np
from typing import Tuple

from algorithms_numpy import USNA
from objective_functions_numpy_online import BaseObjectiveFunction


class USNA_variants(USNA):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        param: np.ndarray,
        objective_function: BaseObjectiveFunction,
        lr_exp: float = 1.0,
        lr_const: float = 1.0,
        lr_add_iter: int = 20,  # No specified in the article
        lr_hess_exp: float = 0.75,
        lr_hess_const: float = 0.1,  # Not specified in the article, and 1.0 diverges
        lr_hess_add_iter: int = 200,  # Not specified, Works better
        averaged: bool = False,  # Whether to use an averaged parameter
        weight_exp: float = 2.0,  # Exponent for the logarithmic weight
        averaged_matrix: bool = False,  # Wether to use an averaged estimate of the inverse hessian
        weight_exp_matrix: float = 2.0,  # Exponent for the logarithmic weight of the averaged inverse hessian
        compute_hessian_param_avg: bool = False,  # If averaged, where to compute the hessian
        generate_Z: str = "canonic",
        sym: bool = True,  # Symmetric estimate of the hessian
        algo: str = "rapport",  # Version of USNA described in the rapport, by default
    ):
        super().__init__(
            param,
            objective_function,
            lr_exp,
            lr_const,
            lr_add_iter,
            lr_hess_exp,
            lr_hess_const,
            lr_hess_add_iter,
            averaged,
            weight_exp,
            averaged_matrix,
            weight_exp_matrix,
            compute_hessian_param_avg,
        )

        self.generate_Z = generate_Z
        self.sym = sym
        self.algo = algo

        self.name += (
            f" {algo}"
            + (f" Z={generate_Z}" if generate_Z != "canonic" else "")
            + ("NS" if not sym else "")
        )

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

    def update_hessian_normal(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a normal random vector, also returns grad
        """
        grad, hessian = self.objective_function.grad_and_hessian(data, self.param)
        Z = np.random.standard_normal(self.param_dim)
        P = self.matrix @ Z
        Q = hessian @ Z
        learning_rate_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * np.dot(Z, Z) <= beta**2:
            product = np.outer(Q, P)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix += -learning_rate_hessian * (product - np.eye(self.param_dim))

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix += -learning_rate_hessian * (product - np.eye(self.param_dim))
                norm = np.linalg.norm(self.matrix, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                beta_prime = 1 / (4 * learning_rate_hessian)
                if norm > beta_prime:
                    self.matrix *= beta_prime / norm

            # Version with added term to ensure positive definiteness like in rapport, but right multiplication
            elif self.algo == "right":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.matrix, Z
                    ) * np.outer(
                        Q, Q
                    )
                else:
                    self.matrix += -learning_rate_hessian * (
                        product - np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * np.einsum(
                        "i,ij,j", Z, self.matrix, Z
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
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        Old version, the multiplication by Z Z^T is done on the RIGHT of the random hessian h_t
        """
        # Generate Z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            # Update the index for the next iteration
            self.k = (self.k + 1) % self.param_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # Compute vectors P and Q
        grad, Q = self.objective_function.grad_and_hessian_column(data, self.param, z)
        # Z is supposed to be sqrt(param_dim) * e_z, but will multiply later after checking <=beta
        P = self.matrix[:, z]
        learning_rate_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * learning_rate_hessian)

        # An alternative to bruno's added term would be the following condition (not studied yet):
        # update_condition =  np.dot(Q, Q) * self.param_dim**2 * P[z] <= beta**2:
        update_condition = np.dot(Q, Q) * self.param_dim**2 <= beta**2
        if update_condition:
            product = self.param_dim * np.outer(Q, P)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix += -learning_rate_hessian * (product - np.eye(self.param_dim))

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix += -learning_rate_hessian * (product - np.eye(self.param_dim))

                # Projection step:
                norm = np.linalg.norm(self.matrix, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                if norm > 1 / (4 * learning_rate_hessian):
                    self.matrix /= norm

            # version with added term to ensure positive definiteness
            elif self.algo == "right":
                if self.sym:
                    self.matrix += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)
                else:
                    self.matrix += -learning_rate_hessian * (
                        product - np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * P[z] * np.outer(Q, Q)

        return grad

    def update_hessian_canonic_left(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        New version, the multiplication by Z Z^T is done on the LEFT of the random hessian h_t
        Also new projected version, close to multiply on left so i put it here, not studied yet
        """
        # Generate Z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.param_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # TODO: we want a line here, not a column. Do a grad_and_hessian_iine function
        # It is the same if random hessians are symmetric, which is the case in our simulations.
        grad, Q = self.objective_function.grad_and_hessian_column(data, self.param, z)
        # Z is supposed to be sqrt(param_dim) * e_z, but will multiply later
        learning_rate_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * learning_rate_hessian)

        if np.dot(Q, Q) * self.param_dim**2 <= beta**2:
            product = self.param_dim * Q.T @ self.matrix

            # rapport version, with the added term and the left multiplication by ZZ^T
            if self.algo == "rapport":
                self.matrix[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.matrix[:, z] += -learning_rate_hessian * product
                self.matrix += (1 + self.sym) * learning_rate_hessian * np.eye(self.param_dim)
                self.matrix[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.matrix, Q
                )

            # Z Z^T replace Id, on top of rapport version
            elif self.algo == "proj":
                product[z] += -self.param_dim
                self.matrix[z, :] += -learning_rate_hessian * product
                if self.sym:
                    self.matrix[:, z] += -learning_rate_hessian * product
                self.matrix[z, z] += learning_rate_hessian**2 * np.einsum(
                    "i,ij,j", Q, self.matrix, Q
                )

        return grad
