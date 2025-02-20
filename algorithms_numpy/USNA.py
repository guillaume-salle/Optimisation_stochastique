import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer, version described in the internship report.
    """

    DEFAULT_LR_EXP = 0.67  # for non-averaged
    CONST_BETA = 1 / 2  # beta_n = CONST_BETA / gamma_n

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int = None,
        batch_size_power: int = 0,
        lr_exp: float = None,  # Not specified in the article for UWASNA, and set to 1.0 for USNA
        lr_const: float = 1.0,  # Set to 1.0 in the article
        lr_add_iter: int = 0,  # No specified in the article, 20 works well except linear regression which prefers 200
        lr_hess_exp: float = 0.75,  # Set to 0.75 in the article
        lr_hess_const: float = 0.1,  # Not specified in the article, and 1.0 diverges
        lr_hess_add_iter: int = 400,  # Not specified, Works better
        averaged: bool = False,  # Whether to use an averaged parameter
        log_weight: float = 2.0,  # Exponent for the logarithmic weight
        multiply_lr_const: bool = False,  # Whether to multiply the step size by the dimension
        multiply_exp: float = None,  # Exponent for the dimension multiplier
        averaged_matrix: bool = False,  # Wether to use an averaged estimate of the inverse hessian
        log_weight_matrix: float = 2.0,  # Exponent for the logarithmic weight of the averaged inverse hessian
        compute_hessian_param_avg: bool = False,  # If averaged, where to compute the hessian
        proj: bool = False,
    ):
        if lr_exp is None:
            lr_exp = 1.0 if not averaged else self.DEFAULT_LR_EXP

        self.name = (
            "USNA"
            + (" AM" if averaged_matrix else "")
            + (" AP" if compute_hessian_param_avg else "")
            + (" P" if proj else "")
            + (f" γ={lr_hess_exp}" if lr_hess_exp != 0.75 else "")
            + (f" c_γ={lr_hess_const}" if lr_hess_const != 0.1 else "")
        )
        self.lr_hess_exp = lr_hess_exp
        self.lr_hess_const = lr_hess_const
        self.lr_hess_add_iter = lr_hess_add_iter
        self.averaged_matrix = averaged_matrix
        self.log_weight_matrix = log_weight_matrix
        self.compute_hessian_param_avg = compute_hessian_param_avg
        self.proj = proj

        self.param_dim = param.shape[0]
        self.matrix = np.eye(param.shape[0])
        self.matrix_not_avg = np.copy(self.matrix) if averaged_matrix else self.matrix
        if averaged_matrix:
            self.sum_weights_matrix = 0

        super().__init__(
            param=param,
            obj_function=obj_function,
            batch_size=batch_size,
            batch_size_power=batch_size_power,
            lr_exp=lr_exp,
            lr_const=lr_const,
            lr_add_iter=lr_add_iter,
            averaged=averaged,
            log_weight=log_weight,
            multiply_lr_const=multiply_lr_const,
            multiply_exp=multiply_exp,
        )

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ):
        """
        Perform one optimization step

        Args:
            data (np.ndarray | Tuple[np.ndarray, np.ndarray]): The input data for the optimization step.
        """
        self.n_iter += 1

        # Update the hessian estimate and get the gradient from intermediate computation
        grad = self.update_hessian(data)
        if self.averaged_matrix:
            self.update_averaged_matrix()

        # Update theta
        learning_rate_theta = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        self.param_not_averaged -= learning_rate_theta * self.matrix @ grad

        if self.averaged:
            self.update_averaged_param()

    def update_hessian(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        """
        # Generate Z, Z is supposed to be sqrt(param_dim) * e_z
        z = np.random.randint(0, self.param_dim)

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

        if np.linalg.norm(hessian_column) * self.param_dim <= self.CONST_BETA / lr_hessian:
            # Compute this product only once and then transpose it
            product = self.param_dim * hessian_column.T @ self.matrix_not_avg

            self.matrix_not_avg[z, :] += -lr_hessian * product
            self.matrix_not_avg[:, z] += -lr_hessian * product
            self.matrix_not_avg[z, z] += lr_hessian**2 * np.einsum(
                "i,ij,j", hessian_column, self.matrix_not_avg, hessian_column
            )
            if self.proj:
                self.matrix_not_avg[z, z] += 2 * lr_hessian * self.param_dim
            else:
                self.matrix_not_avg += 2 * lr_hessian * np.eye(self.param_dim)

        return grad

    def update_averaged_matrix(self) -> None:
        """
        Update the averaged condition matrix using the current matrix and the sum of weights.
        """
        if self.log_weight_matrix > 0:
            weight_matrix = np.log(self.n_iter + 1) ** self.log_weight_matrix
        else:
            weight_matrix = 1
        self.sum_weights_matrix += weight_matrix
        self.matrix += (weight_matrix / self.sum_weights_matrix) * (
            self.matrix_not_avg - self.matrix
        )
