import numpy as np
import math
from typing import Tuple

from algorithms_numpy import USNA, BaseOptimizer
from objective_functions_numpy.streaming import BaseObjectiveFunction


class USNA_variants(USNA):
    """
    Universal Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        mini_batch: int = None,
        mini_batch_power: float = 0.0,
        lr_exp: float = None,
        lr_const: float = BaseOptimizer.DEFAULT_LR_CONST,
        lr_add_iter: int = BaseOptimizer.DEFAULT_LR_ADD_ITER,
        averaged: bool = False,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        multiply_lr: float = BaseOptimizer.DEFAULT_MULTIPLY_LR,
        # USNA specific parameters
        lr_hess_exp: float = 0.75,
        lr_hess_const: float = 0.1,
        lr_hess_add_iter: int = 400,
        averaged_matrix: bool = False,
        log_weight_matrix: float = 2.0,
        compute_hessian_param_avg: bool = False,
        proj: bool = False,
        # New parameters for the variants
        generate_Z: str = "canonic",
        sym: bool = True,  # Symmetric estimate of the hessian
        multiply_between: bool = False,  # If Z Z^T is placed between the product A H_n, or outside
        algo: str = "rapport",  # Version of USNA described in the rapport, by default
    ):
        super().__init__(
            param=param,
            obj_function=obj_function,
            mini_batch=mini_batch,
            mini_batch_power=mini_batch_power,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add_iter=lr_add_iter,
            averaged=averaged,
            log_weight=log_weight,
            multiply_lr=multiply_lr,
            lr_hess_exp=lr_hess_exp,
            lr_hess_const=lr_hess_const,
            lr_hess_add_iter=lr_hess_add_iter,
            averaged_matrix=averaged_matrix,
            log_weight_matrix=log_weight_matrix,
            compute_hessian_param_avg=compute_hessian_param_avg,
            proj=proj,
        )

        self.generate_Z = generate_Z
        self.sym = sym
        self.multiply_between = multiply_between
        self.algo = algo

        self.name += (
            f" {algo}"
            + (f" Z={generate_Z}" if generate_Z != "canonic" else "")
            + (" NS" if not sym else "")
            + (" B" if multiply_between else "")
        )

        # Test different versions of the algorithm
        versions = [
            "article",
            "article_v2",
            "rapport",
        ]
        if algo not in versions:
            raise ValueError("Invalid value for algo. Choose " + ", ".join(versions) + ".")

        # In case we know Z is a random vector of canonic base, we can compute faster P and Q
        if generate_Z == "normal":
            self.update_hessian = self.update_hessian_normal
        elif generate_Z in ["canonic", "canonic deterministic"]:
            if multiply_between:
                self.update_hessian = self.update_hessian_canonic_between
            else:
                self.update_hessian = self.update_hessian_canonic_outside
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
        # Compute grad in the NOT averaged param, and the whole hessian in the desired param
        grad, hessian = self.obj_function.grad_and_hessian(data, self.param)
        if self.compute_hessian_param_avg:
            hessian = self.obj_function.hessian(data, self.param)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian = self.obj_function.grad_and_hessian_column(data, self.param_not_averaged)

        # Generate Z, normalize and multiply by sqrt(d)
        Z = np.random.standard_normal(self.param_dim)
        Z *= math.sqrt(self.param_dim) / np.linalg.norm(Z)

        H_Z = hessian @ Z
        lr_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )

        # Compute beta_n, the threshold for the update
        if self.algo == "article" or self.algo == "article_v2":
            # To respect conditions beta = gamma - 1/2, and gamma_0 beta_0 = 1
            beta_n = self.n_iter ** (self.lr_hess_exp - 0.5) / self.lr_hess_const
        else:
            beta_n = self.CONST_BETA / lr_hessian

        # Update condition : ||H Z|| * ||Z|| <= beta_n
        if np.linalg.norm(H_Z) * math.sqrt(self.param_dim) <= beta_n:
            # Compute product once, then transpose it
            if self.multiply_between:
                # product := H Z Z^T A
                A_Z = self.matrix_not_avg @ Z
                product = np.outer(H_Z, A_Z)
            else:
                # product := Z Z^T H A
                A_H_Z = self.matrix_not_avg @ H_Z
                product = np.outer(Z, A_H_Z)  # Assuming h and A are symmetric

            # Term to add, Id or Z Z^T
            I_d = np.eye(self.param_dim) if not self.proj else np.outer(Z, Z)

            # Article version before revision
            # A_new := A_old - lr_hessian * (H Z Z^T A_old + A_old Z Z^T H - 2 Id)
            if self.algo == "article" or self.algo == "article_v2":
                if self.sym:
                    self.matrix_not_avg += -lr_hessian * (product + product.transpose() - 2 * I_d)
                else:
                    # We multiply learning rate by 2 to match the symmetric case
                    self.matrix_not_avg += -2 * lr_hessian * (product - I_d)

                # Article version after revision, with projection step
                if self.algo == "article_v2":
                    norm = np.linalg.norm(self.matrix_not_avg, ord="fro")
                    # we take beta'_n := gamma_0 * n^(1 - gamma)
                    beta_prime = self.lr_hess_const * self.n_iter ** (1 - self.lr_hess_exp)
                    if norm > beta_prime:
                        self.matrix_not_avg *= beta_prime / norm

            # Version with added term to ensure positive definiteness and left multiplication for h, like in rapport
            # A_new := A_old - lr_hessian * (Z Z.T H A_old + A_old H Z Z.T - 2 Id) + lr_hessian^2 * Z Z.T H A_old H Z Z.T
            elif self.algo == "rapport":
                A_H_Z = np.matmul(self.matrix_not_avg, H_Z)
                Z_Zt_H_A = np.outer(Z, A_H_Z)  # Assuming h and A are symmetric
                if self.sym:
                    self.matrix_not_avg += -lr_hessian * (
                        Z_Zt_H_A + Z_Zt_H_A.transpose() - 2 * I_d
                    ) + lr_hessian**2 * np.outer(Z_Zt_H_A @ H_Z, Z)
                else:
                    # We multiply learning rate by 2 to match the symmetric case
                    self.matrix_not_avg += -2 * lr_hessian * (Z_Zt_H_A - I_d) + (
                        2 * lr_hessian
                    ) ** 2 * np.dot(H_Z, A_H_Z) * np.outer(Z_Zt_H_A @ H_Z, Z)

        return grad

    # Different function because if Z is a canonic vector, some computations can a be simple column/coeff selection
    def update_hessian_canonic_between(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic vector, also returns grad
        Old version, the multiplication by Z Z^T is done on the RIGHT of the random hessian h_t
        """
        # Generate Z, Z is supposed to be sqrt(param_dim) * e_z, but we multiply by param_dim directly later
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            # Update the index for the next iteration
            self.k = (self.k + 1) % self.param_dim
        else:
            raise ValueError(
                "Invalid value for 'generate_Z'. Choose 'canonic' or 'canonic deterministic'."
            )

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        if self.compute_hessian_param_avg:
            hessian_column = self.obj_function.hessian_column(data, self.param, z)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_column = self.obj_function.grad_and_hessian_column(
                data, self.param_not_averaged, z
            )

        lr_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )

        # Compute beta_n, the threshold for the update
        if self.algo == "article" or self.algo == "article_v2":
            # To respect conditions beta = gamma - 1/2, and gamma_0 beta_0 = 1
            beta_n = self.n_iter ** (self.lr_hess_exp - 0.5) / self.lr_hess_const
        else:
            beta_n = self.CONST_BETA / lr_hessian

        # Update condition : ||H Z|| * ||Z|| <= beta_n
        if np.linalg.norm(hessian_column) * self.param_dim <= beta_n:
            matrix_line = self.matrix_not_avg[:, z]
            product = self.param_dim * np.outer(hessian_column, matrix_line)

            if self.sym:
                self.matrix_not_avg += -lr_hessian * (product + product.transpose())
            else:
                self.matrix_not_avg += -2 * lr_hessian * product
            if self.proj:
                self.matrix_not_avg[z, z] += 2 * lr_hessian * self.param_dim
            else:
                self.matrix_not_avg += 2 * lr_hessian * np.eye(self.param_dim)

            # Article version before revision, do nothing
            if self.algo == "article":
                pass

            # Article version after revision, with projection step
            elif self.algo == "article v2":
                norm = np.linalg.norm(self.matrix_not_avg, ord="fro")
                # we take beta'_n := gamma_0 * n^(1 - gamma)
                beta_prime = self.lr_hess_const * self.n_iter ** (1 - self.lr_hess_exp)
                if norm > beta_prime:
                    self.matrix_not_avg *= beta_prime / norm

            # versions with added term to ensure positive definiteness, and without projection
            elif self.algo == "rapport":
                self.matrix_not_avg += (
                    lr_hessian**2 * matrix_line[z] * np.outer(hessian_column, hessian_column)
                )

        return grad

    def update_hessian_canonic_outside(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic vector, also returns grad
        The multiplication by Z Z^T is done on the LEFT of the random hessian h_t : A Z Z^T h_t
        """
        # Generate Z, Z is supposed to be sqrt(param_dim) * e_z
        if self.generate_Z == "canonic":
            z = np.random.randint(0, self.param_dim)
        elif self.generate_Z == "canonic deterministic":
            z = self.k
            self.k = (self.k + 1) % self.param_dim

        # Compute grad in the NOT averaged param, and hessian column in the desired param
        if self.compute_hessian_param_avg:
            hessian_column = self.obj_function.hessian_column(data, self.param, z)
            grad = self.obj_function.grad(data, self.param_not_averaged)
        else:
            grad, hessian_column = self.obj_function.grad_and_hessian_column(
                data, self.param_not_averaged, z
            )

        lr_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )

        # Compute beta_n, the threshold for the update
        if self.algo == "article" or self.algo == "article_v2":
            # To respect conditions beta = gamma - 1/2, and gamma_0 beta_0 = 1
            beta_n = self.n_iter ** (self.lr_hess_exp - 0.5) / self.lr_hess_const
        else:
            beta_n = self.CONST_BETA / lr_hessian

        # Update condition : ||H Z|| * ||Z|| <= beta_n
        if np.linalg.norm(hessian_column) * self.param_dim <= beta_n:
            product = self.param_dim * hessian_column.T @ self.matrix_not_avg

            if self.sym:
                self.matrix_not_avg[z, :] += -lr_hessian * product
                self.matrix_not_avg[:, z] += -lr_hessian * product
            else:
                self.matrix_not_avg[z, :] += -2 * lr_hessian * product

            if self.proj:
                self.matrix_not_avg[z, z] += 2 * lr_hessian * self.param_dim
            else:
                self.matrix_not_avg += 2 * lr_hessian * np.eye(self.param_dim)

            # Article version before revision, do nothing
            if self.algo == "article":
                pass

            # Article version after revision, with projection step
            elif self.algo == "article v2":
                norm = np.linalg.norm(self.matrix_not_avg, ord="fro")
                # we take beta'_n := gamma_0 * n^(1 - gamma)
                beta_prime = self.lr_hess_const * self.n_iter ** (1 - self.lr_hess_exp)
                if norm > beta_prime:
                    self.matrix_not_avg *= beta_prime / norm

            # version with the added term to ensure positive definiteness
            elif self.algo == "rapport":
                self.matrix_not_avg[z, z] += lr_hessian**2 * np.einsum(
                    "i,ij,j", hessian_column, self.matrix_not_avg, hessian_column
                )

        return grad
