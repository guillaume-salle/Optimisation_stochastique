import numpy as np
import math
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
        lr_exp: float = None,  # Not specified in the article for UWASNA, and set to 1.0 for USNA
        lr_const: float = 1.0,
        lr_add_iter: int = 0,  # No specified in the article
        lr_hess_exp: float = 0.75,
        lr_hess_const: float = 0.1,  # Not specified in the article, and 1.0 diverges
        lr_hess_add_iter: int = 0,  # Not specified, Works better
        averaged: bool = False,  # Whether to use an averaged parameter
        log_weight: float = 2.0,  # Exponent for the logarithmic weight
        averaged_matrix: bool = False,  # Wether to use an averaged estimate of the inverse hessian
        log_weight_matrix: float = 2.0,  # Exponent for the logarithmic weight of the averaged inverse hessian
        compute_hessian_param_avg: bool = False,  # If averaged, where to compute the hessian
        generate_Z: str = "canonic",
        sym: bool = True,  # Symmetric estimate of the hessian
        multiply_right: bool = False,  # If the random hessian h is multiplied on the right by Z Z^T
        proj: bool = False,  # If the identity is replaced by Z Z^T, to reduce variance
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
            log_weight,
            averaged_matrix,
            log_weight_matrix,
            compute_hessian_param_avg,
        )

        self.generate_Z = generate_Z
        self.sym = sym
        self.multiply_right = multiply_right
        self.proj = proj
        self.algo = algo

        self.name += (
            f" {algo}"
            + (f" Z={generate_Z}" if generate_Z != "canonic" else "")
            + (" NS" if not sym else "")
            + (" R" if multiply_right else "")
            + (" P" if proj else "")
        )

        # Test different versions of the algorithm
        versions = [
            "article",
            "article_v2",
            "rapport",
            # "new", # TODO
        ]
        if algo not in versions:
            raise ValueError("Invalid value for algo. Choose " + ", ".join(versions) + ".")

        # In case we know Z is a random vector of canonic base, we can compute faster P and Q
        if generate_Z == "normal":
            self.update_hessian = self.update_hessian_normal
        elif generate_Z in ["canonic", "canonic deterministic"]:
            if multiply_right:
                self.update_hessian = self.update_hessian_canonic_right
            else:
                self.update_hessian = self.update_hessian_canonic_left
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
        # Compute grad in the NOT averaged param, and hessian in the desired param
        grad, hessian = self.objective_function.grad_and_hessian(data, self.param)
        if self.compute_hessian_param_avg:
            hessian = self.objective_function.hessian(data, self.param)
            grad = self.objective_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian = self.objective_function.grad_and_hessian_column(
                data, self.param_not_averaged
            )

        # Generate Z, normalize and multiply by sqrt(d)
        Z = np.random.standard_normal(self.param_dim)
        Z *= math.sqrt(self.param_dim) / np.linalg.norm(Z)

        h_Z = hessian @ Z
        learning_rate_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * learning_rate_hessian)

        if np.linalg.norm(h_Z) <= beta:
            I_d = np.eye(self.param_dim) if not self.proj else np.outer(Z, Z)

            if self.multiply_right:
                # product := h Z Z^T A
                A_Z = self.matrix_not_avg @ Z
                Zt_A = np.matmul(Z, self.matrix_not_avg)
                product = np.outer(h_Z, Zt_A)
            else:
                # product := Z Z^T h A
                A_h_Z = np.matmul(self.matrix_not_avg, h_Z)
                product = np.outer(Z, A_h_Z)  # Assuming h and A are symmetric

            # Article version before revision
            # A_new := A_old - lr_hessian * (h Z Z^T A_old + A_old Z Z^T h - 2 Id)
            if self.algo == "article":
                if self.sym:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * I_d
                    )
                else:
                    # We multiply by 2 to match the symmetric case
                    self.matrix_not_avg += -2 * learning_rate_hessian * (product - I_d)

            # Article version after revision, with projection step
            elif self.algo == "article_v2":
                A_Z = self.matrix_not_avg @ Z
                h_Z_Zt_A = np.outer(h_Z, A_Z)
                if self.sym:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        h_Z_Zt_A + h_Z_Zt_A.transpose() - 2 * I_d
                    )
                else:
                    self.matrix_not_avg += -2 * learning_rate_hessian * (h_Z_Zt_A - I_d)

                # Projection step
                norm = np.linalg.norm(self.matrix_not_avg, ord="fro")
                # we take beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                beta_prime = 1 / (4 * learning_rate_hessian)
                if norm > beta_prime:
                    self.matrix_not_avg *= beta_prime / norm

            # Version with added term to ensure positive definiteness and left multiplication for h, like in rapport
            # A_new := A_old - lr_hessian * (Z Z.T h A_old + A_old h Z Z.T - 2 Id) + lr_hessian^2 * Z Z.T h A_old h Z Z.T
            elif self.algo == "rapport":
                A_h_Z = np.matmul(self.matrix_not_avg, h_Z)
                Z_Zt_h_A = np.outer(Z, A_h_Z)  # Assuming h and A are symmetric
                Z_Zt = I_d if self.proj else np.outer(Z, Z)
                if self.sym:
                    self.matrix_not_avg += (
                        -learning_rate_hessian * (Z_Zt_h_A + Z_Zt_h_A.transpose() - 2 * I_d)
                        + learning_rate_hessian**2 * np.dot(h_Z, A_h_Z) * Z_Zt
                    )
                else:
                    self.matrix_not_avg += (
                        -2 * learning_rate_hessian * (Z_Zt_h_A - I_d)
                        + learning_rate_hessian**2 * np.dot(h_Z, A_h_Z) * Z_Zt
                    )

            # Version like in rapport, but right multiplication for h by Z Z^T
            # A_new := A_old - lr_hessian * (h Z Z.T A_old + A_old Z Z.T h - 2 Id) + lr_hessian^2 * h Z Z.T A_old Z Z.T h
            elif self.algo == "rapport_right":
                A_Z = self.matrix_not_avg @ Z
                h_Z_Zt_A = np.outer(h_Z, A_Z)
                Z_Zt = I_d if self.proj else np.outer(Z, Z)
                if self.sym:
                    self.matrix_not_avg += (
                        -learning_rate_hessian
                        * (h_Z_Zt_A + h_Z_Zt_A.transpose() - 2 * np.eye(self.param_dim))
                        + learning_rate_hessian**2 * np.dot(Z, A_Z) * Z_Zt
                    )
                else:
                    self.matrix_not_avg += (
                        -learning_rate_hessian * 2 * (h_Z_Zt_A - np.eye(self.param_dim))
                        + learning_rate_hessian**2 * np.dot(Z, A_Z) * Z_Zt
                    )

        return grad

    # Different function because if Z is a canonic vector, some computations can a be simple column/coeff selection
    def update_hessian_canonic_right(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        Old version, the multiplication by Z Z^T is done on the RIGHT of the random hessian h_t
        """
        # Generate Z, Z is supposed to be sqrt(param_dim) * e_z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            # Update the index for the next iteration
            self.k = (self.k + 1) % self.param_dim
        else:
            raise ValueError("Invalid value for Z. Choose 'canonic' or 'canonic deterministic'.")

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        if self.compute_hessian_param_avg:
            hessian_column = self.objective_function.hessian_column(data, self.param, z)
            grad = self.objective_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_column = self.objective_function.grad_and_hessian_column(
                data, self.param_not_averaged, z
            )

        matrix_line = self.matrix_not_avg[:, z]
        learning_rate_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * learning_rate_hessian)

        # An alternative to bruno's added term would be the following condition (not studied yet):
        # update_condition =  np.dot(Q, Q) * self.param_dim**2 * P[z] <= beta**2:
        update_condition = np.dot(hessian_column, hessian_column) * self.param_dim**2 <= beta**2
        if update_condition:
            product = self.param_dim * np.outer(hessian_column, matrix_line)

            # Article version before revision
            if self.algo == "article":
                if self.sym:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.param_dim)
                    )

            # Article version after revision
            elif self.algo == "article v2":
                if self.sym:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    )
                else:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.param_dim)
                    )

                # Projection step:
                norm = np.linalg.norm(self.matrix_not_avg, ord="fro")
                # beta'_n := gamma_n / (beta_n)^2 = 1 / (4 * learning_rate_hessian)
                if norm > 1 / (4 * learning_rate_hessian):
                    self.matrix_not_avg /= norm

            # version with added term to ensure positive definiteness
            elif self.algo == "right":
                if self.sym:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product + product.transpose() - 2 * np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * matrix_line[z] * np.outer(
                        hessian_column, hessian_column
                    )
                else:
                    self.matrix_not_avg += -learning_rate_hessian * (
                        product - np.eye(self.param_dim)
                    ) + learning_rate_hessian**2 * matrix_line[z] * np.outer(
                        hessian_column, hessian_column
                    )

        return grad

    def update_hessian_canonic_left(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        The multiplication by Z Z^T is done on the LEFT of the random hessian h_t
        Also new projected version, close to multiply on left so i put it here, not studied yet
        """
        # Generate Z, Z is supposed to be sqrt(param_dim) * e_z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.param_dim

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        # TODO: we want a line here, not a column. Do a grad_and_hessian_line function
        # It is the same if random hessians are symmetric, which is the case in our simulations.
        if self.compute_hessian_param_avg:
            hessian_column = self.objective_function.hessian_column(data, self.param, z)
            grad = self.objective_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_column = self.objective_function.grad_and_hessian_column(
                data, self.param_not_averaged, z
            )

        lr_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * lr_hessian)

        if np.dot(hessian_column, hessian_column) * self.param_dim**2 <= beta**2:
            product = self.param_dim * hessian_column.T @ self.matrix_not_avg

            if self.algo == "article":
                pass

            elif self.algo == "article v2":
                pass

            # rapport version, with the added term and the left multiplication by ZZ^T
            elif self.algo == "rapport":
                self.matrix_not_avg[z, :] += -lr_hessian * product
                if self.sym:
                    self.matrix_not_avg[:, z] += -lr_hessian * product
                if self.proj:
                    self.matrix_not_avg[z, z] += (1 + self.sym) * lr_hessian * self.param_dim
                else:
                    self.matrix_not_avg += (1 + self.sym) * lr_hessian * np.eye(self.param_dim)
                self.matrix_not_avg[z, z] += lr_hessian**2 * np.einsum(
                    "i,ij,j", hessian_column, self.matrix_not_avg, hessian_column
                )

        return grad
