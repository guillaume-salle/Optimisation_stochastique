import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class USNA(BaseOptimizer):
    """
    Universal Stochastic Newton Algorithm optimizer, version described in the internship report.
    """

    class_name = "USNA"

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
    ):
        self.name = (
            "U"
            + ("W" if averaged and weight_exp != 0.0 else "")
            + ("A" if averaged else "")
            + "SNA"
            + " AM" * averaged_matrix
            + " AP" * compute_hessian_param_avg
            + (f" α={lr_exp}")
            + (f" γ={lr_hess_exp}" if lr_hess_exp != 0.75 else "")
        )
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter
        self.lr_hess_exp = lr_hess_exp
        self.lr_hess_const = lr_hess_const
        self.lr_hess_add_iter = lr_hess_add_iter
        self.averaged_matrix = averaged_matrix
        self.weight_exp_matrix = weight_exp_matrix
        self.compute_hessian_param_avg = compute_hessian_param_avg

        self.matrix = np.eye(param.shape[0])
        self.matrix_not_avg = np.copy(self.matrix) if averaged_matrix else self.matrix
        self.param_dim = param.shape[0]

        super().__init__(param, objective_function, averaged, weight_exp)

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

        # Update theta
        learning_rate_theta = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        self.param -= learning_rate_theta * self.matrix @ grad

        self.update_averaged_param()
        self.update_averaged_matrix()

    def update_hessian(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """
        Update the hessian estimate with a canonic random vector, also returns grad
        """
        # Generate Z
        z = np.random.randint(0, self.param_dim)

        # TODO: we want a line here, not a column. Do a grad_and_hessian_line function
        # (It is the same if random hessians are symmetric, which is the case in our simulations.)
        grad, Q = self.objective_function.grad_and_hessian_column(data, self.param, z)
        # Z is supposed to be sqrt(param_dim) * e_z, but will multiply later
        lr_hessian = self.lr_hess_const * (self.n_iter + self.lr_hess_add_iter) ** (
            -self.lr_hess_exp
        )
        beta = 1 / (2 * lr_hessian)

        if np.linalg.norm(Q) * self.param_dim <= beta:
            product = self.param_dim * Q.T @ self.matrix

            self.matrix[z, :] += -lr_hessian * product
            self.matrix[:, z] += -lr_hessian * product
            self.matrix += 2 * lr_hessian * np.eye(self.param_dim)
            self.matrix[z, z] += lr_hessian**2 * np.einsum("i,ij,j", Q, self.matrix, Q)

        return grad

    def update_averaged_matrix(self) -> None:
        """
        Update the averaged condition matrix using the current matrix and the sum of weights.
        """

        if not self.averaged_matrix:
            return

        if self.weight_exp_matrix > 0:
            weight = np.log(self.n_iter + 1) ** self.weight_exp_matrix
        else:
            weight = 1
        self.sum_weights += weight
        self.matrix += (weight / self.sum_weights) * (self.matrix_not_avg - self.matrix)
