import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy.streaming import BaseObjectiveFunction


class SNA(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer

    SNA specific parameters:
    identity_weight (int): Weight for the initial identity matrix.
    compute_hessian_param_avg (bool): If averaged, where to compute the hessian.
    compute_inverse (bool): Actually compute the inverse, or just solve the system.
    sherman_morrison (bool): Whether to use the Sherman-Morrison formula.
    """

    name = "SNA"

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int = None,
        batch_size_power: float = 0.0,
        lr_exp: float = None,
        lr_const: float = BaseOptimizer.DEFAULT_LR_CONST,
        lr_add_iter: int = None,
        averaged: bool = False,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        multiply_lr: float | str = 0.0,
        # SNA specific parameters :
        identity_weight: int = 400,
        compute_hessian_param_avg: bool = False,
        compute_inverse: bool = False,
        sherman_morrison: bool = True,
    ):
        if compute_hessian_param_avg:
            self.name += " AP"  # AP = Averaged Parameter
        self.identity_weight = identity_weight
        self.compute_hessian_param_avg = compute_hessian_param_avg
        self.compute_inverse = compute_inverse
        self.sherman_morrison = sherman_morrison

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
            multiply_lr=multiply_lr,
        )

        # For batch_size=1 we can use Sherman-Morrison formula if available
        if self.batch_size == 1 and sherman_morrison and hasattr(obj_function, "sherman_morrison"):
            self.name += " SM"
            self.step = self.step_sherman_morrison
            self.hessian_inv = np.eye(param.shape[0])
        else:
            self.hessian_bar = np.eye(param.shape[0])
            if compute_inverse:
                self.hessian_inv = np.eye(param.shape[0])

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
            grad = self.obj_function.grad(data, self.param_not_averaged)
            hessian = self.obj_function.hessian(data, self.param)
        else:  # faster, allow to re-use the grad from hessian computation
            grad, hessian = self.obj_function.grad_and_hessian(data, self.param_not_averaged)

        # Update the running average of the Hessian
        n_matrix = self.n_iter + self.identity_weight
        self.hessian_bar += (hessian - self.hessian_bar) / n_matrix

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        if self.multiply_lr and self.batch_size > 1:
            learning_rate = min(learning_rate, self.expected_first_lr)
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
            grad = self.obj_function.grad(data, self.param_not_averaged)
            # n_iter + lr_add_iter ??
            sherman_morrison = self.obj_function.sherman_morrison(data, self.param, self.n_iter)
        else:  # faster, allow to re-use the grad from hessian computation
            grad, sherman_morrison = self.obj_function.grad_and_sherman_morrison(
                data, self.param_not_averaged, self.n_iter  # n_iter + lr_add_iter ??
            )

        # Update the inverse Hessian matrix using the Sherman-Morrison equation
        n_matrix = self.n_iter + self.identity_weight
        product = np.dot(self.hessian_inv, sherman_morrison)
        self.hessian_inv += (1 / (n_matrix - 1)) * (
            self.hessian_inv
            - n_matrix
            / (n_matrix - 1 + np.dot(sherman_morrison, product))
            * np.outer(product, product)
        )

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        self.param_not_averaged -= learning_rate * self.hessian_inv @ grad

        if self.averaged:
            self.update_averaged_param()
