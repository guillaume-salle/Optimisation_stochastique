import numpy as np
import math
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class SNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    Uses a learning rate lr = lr_const * (n_iter + lr_add_iter)^(-lr_exp) for optimization.
    Averaged parameter can be calculated with a logarithmic weight, i.e. the weight is
    calculated as log(n_iter+1)^weight_exp.

    Parameters:
    param (np.ndarray): Initial parameters for the optimizer.
    lr_exp (float): Exponent for learning rate decay.
    lr_const (float): Constant multiplier for learning rate.
    lr_add_iter (int): Additional iterations for learning rate calculation.
    averaged (bool): Whether to use an averaged parameter.
    weight_exp (float): Exponent for the logarithmic weight.
    """

    def __init__(
        self,
        param: np.ndarray,
        objective_function: BaseObjectiveFunction,
        lr_exp: float = None,  # Not specified in the article for WASNA, and set to 1.0 for SNA
        lr_const: float = 1.0,
        lr_add_iter: int = 0,
        identity_weight: int = 400,  # Weight more the initial identity matrix
        averaged: bool = False,  # Whether to use an averaged parameter
        log_weight: float = 2.0,  # Exponent for the logarithmic weight
        compute_hessian_param_avg: bool = False,  # If averaged, where to compute the hessian
        compute_inverse: bool = False,  # Actually compute inverse, or just solve the system
        sherman_morrison: bool = True,  # Whether to use the Sherman-Morrison formula
    ):
        if lr_exp is None:
            lr_exp = 1.0 if not averaged else 0.75
        self.name = (
            ("W" if averaged and log_weight != 0.0 else "")
            + ("A" if averaged else "")
            + "SNA"
            + (f" α={lr_exp}")
            + (f" τ_theta={log_weight}" if log_weight != 2.0 and log_weight != 0.0 else "")
            + (" AP" if compute_hessian_param_avg else "")  # AP = Averaged Parameter
        )
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter
        self.identity_weight = identity_weight
        self.compute_hessian_param_avg = compute_hessian_param_avg
        self.compute_inverse = compute_inverse
        self.sherman_morrison = sherman_morrison

        if sherman_morrison and hasattr(objective_function, "sherman_morrison"):
            self.step = self.step_sherman_morrison
            self.hessian_inv = np.eye(param.shape[0])
            self.name += " SM"
        else:
            self.hessian_bar = np.eye(param.shape[0])
            if compute_inverse:
                self.hessian_inv = np.eye(param.shape[0])

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
        # Compute the gradient and Hessian of the objective function
        if self.compute_hessian_param_avg:  # cf article
            grad = self.objective_function.grad(data, self.param_not_averaged)
            hessian = self.objective_function.hessian(data, self.param)
        else:  # faster, allow to re-use the grad from hessian computation
            grad, hessian = self.objective_function.grad_and_hessian(data, self.param_not_averaged)

        # Update the running average of the Hessian
        n_matrix = self.n_iter + self.identity_weight
        self.hessian_bar = ((n_matrix - 1) / n_matrix) * self.hessian_bar + hessian / n_matrix

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        if self.compute_inverse:
            self.hessian_inv = np.linalg.inv(self.hessian_bar)
            self.param_not_averaged -= learning_rate * self.hessian_inv @ grad
        else:  # faster and more stable, no need to compute the whole inverse
            self.param_not_averaged -= learning_rate * np.linalg.solve(self.hessian_bar, grad)

        if self.averaged:
            self.update_averaged_param()

    def step_sherman_morrison(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ):
        """
        Perform one optimization step using the Sherman-Morrison formula

        Args:
            data (np.ndarray | Tuple[np.ndarray, np.ndarray]): The input data for the optimization step.
        """
        self.n_iter += 1
        if self.compute_hessian_param_avg:  # cf article
            grad = self.objective_function.grad(data, self.param_not_averaged)
            sherman_morrison = self.objective_function.sherman_morrison(data, self.param)
        else:  # faster, allow to re-use the grad from hessian computation
            grad, sherman_morrison = self.objective_function.grad_and_sherman_morrison(
                data, self.param_not_averaged
            )

        # Update the inverse Hessian matrix using the Sherman-Morrison equation
        n_matrix = self.n_iter + self.identity_weight
        product = np.dot(self.hessian_inv, sherman_morrison) / n_matrix
        self.hessian_inv -= (1 / n_matrix) * self.hessian_inv + np.outer(
            product, product / (n_matrix ** (-1) + np.dot(sherman_morrison, product))
        )

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        self.param_not_averaged -= learning_rate * self.hessian_inv @ grad

        if self.averaged:
            self.update_averaged_param()
