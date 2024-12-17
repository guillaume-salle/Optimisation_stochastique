import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer, version described in the internship report.
    """

    def __init__(
        self,
        param: np.ndarray,
        objective_function: BaseObjectiveFunction,
        lr_exp: float = None,  # Not specified in the article for UWASNA, and set to 1.0 for USNA
        lr_const: float = 1.0,  # Set to 1.0 in the article
        lr_add_iter: int = 0,  # No specified in the article, 20 works well except linear regression which prefers 200
        lr_hess_exp: float = 0.75,  # Set to 0.75 in the article
        lr_hess_const: float = 0.1,  # Not specified in the article, and 1.0 diverges
        lr_hess_add_iter: int = 400,  # Not specified, Works better
        averaged: bool = False,  # Whether to use an averaged parameter
        log_weight: float = 2.0,  # Exponent for the logarithmic weight
        averaged_matrix: bool = False,  # Wether to use an averaged estimate of the inverse hessian
        log_weight_matrix: float = 2.0,  # Exponent for the logarithmic weight of the averaged inverse hessian
        compute_hessian_param_avg: bool = False,  # If averaged, where to compute the hessian
    ):
        if lr_exp is None:
            lr_exp = 1.0 if not averaged else 0.75

        self.name = (
            "U"
            + ("W" if averaged and log_weight != 0.0 else "")
            + ("A" if averaged else "")
            + "SNA"
            + " AM" * averaged_matrix
            + " AP" * compute_hessian_param_avg
            + (f" α={lr_exp}")
            + (f" c_α={lr_const}" if lr_const != 1.0 else "")
            + (f" γ={lr_hess_exp}" if lr_hess_exp != 0.75 else "")
        )
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter
        self.lr_hess_exp = lr_hess_exp
        self.lr_hess_const = lr_hess_const
        self.lr_hess_add_iter = lr_hess_add_iter
        self.averaged_matrix = averaged_matrix
        self.log_weight_matrix = log_weight_matrix
        self.compute_hessian_param_avg = compute_hessian_param_avg

        self.matrix = np.eye(param.shape[0])
        self.matrix_not_avg = np.copy(self.matrix) if averaged_matrix else self.matrix
        self.param_dim = param.shape[0]

        super().__init__(param, objective_function, averaged, log_weight)

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
        # TODO: we want a line here, not a column. Do a grad_and_hessian_line function
        # (It is the same if random hessians are symmetric, which is the case in our simulations.)
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

        # if np.linalg.norm(Q) * self.param_dim <= beta:
        if np.dot(hessian_column, hessian_column) * self.param_dim**2 <= beta**2:
            # Compute this product only once and then transpose it
            product = self.param_dim * hessian_column.T @ self.matrix_not_avg

            self.matrix_not_avg[z, :] += -lr_hessian * product
            self.matrix_not_avg[:, z] += -lr_hessian * product
            self.matrix_not_avg += 2 * lr_hessian * np.eye(self.param_dim)
            self.matrix_not_avg[z, z] += lr_hessian**2 * np.einsum(
                "i,ij,j", hessian_column, self.matrix_not_avg, hessian_column
            )

        return grad

    def update_averaged_matrix(self) -> None:
        """
        Update the averaged condition matrix using the current matrix and the sum of weights.
        """
        if self.log_weight_matrix > 0:
            weight = np.log(self.n_iter + 1) ** self.log_weight_matrix
        else:
            weight = 1
        self.sum_weights += weight
        self.matrix += (weight / self.sum_weights) * (self.matrix_not_avg - self.matrix)
