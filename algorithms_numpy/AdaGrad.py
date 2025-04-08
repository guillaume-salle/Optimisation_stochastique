import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy.streaming import BaseObjectiveFunction


class AdaGrad(BaseOptimizer):
    """
    Adagrad optimizer. Uses a learning rate lr = lr_const * (n_iter + lr_add_iter)^(-lr_exp) for optimization.

    AdaGrad specific parameters:
    epsilon (float): Small constant to avoid singularity problems.
    true_adagrad (bool): Whether to use the true Adagrad update rule, or one with a decreasing learning rate.
    """

    name = "Ada"

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int = None,
        batch_size_power: float = 0.0,
        lr_exp: float = None,
        lr_const: float = BaseOptimizer.DEFAULT_LR_CONST,
        lr_add_iter: int = None,
        averaged: bool = None,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        multiply_lr: float = 0.0,
        # AdaGrad specific parameter
        epsilon: float = 1e-8,
        true_ada: bool = True,
    ):
        # Initialize the sum of the gradients
        self.sum_grad_sq = np.zeros_like(param) + epsilon
        self.true_ada = true_ada
        self.name += " F" if not true_ada else ""

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

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ):
        """
        Perform one optimization step

        Parameters:
        data (np.ndarray | Tuple[np.ndarray, np.ndarray]): The input data for the optimization step.
        """
        self.n_iter += 1
        grad = self.obj_function.grad(data, self.param_not_averaged)

        # Update the sum of the gradients
        self.sum_grad_sq += grad**2

        # Update the non averaged parameter
        if self.true_ada:
            # Constant learning, same as the first iteration for comparison
            learning_rate = self.lr_const * (1 + self.lr_add_iter) ** (-self.lr_exp)
        else:
            learning_rate = (
                self.lr_const
                * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
                * np.sqrt(self.n_iter)
            )
        if self.multiply_lr and self.batch_size > 1:
            learning_rate = min(learning_rate, self.expected_first_lr)
        self.param_not_averaged -= learning_rate * grad / np.sqrt(self.sum_grad_sq)

        if self.averaged:
            self.update_averaged_param()
