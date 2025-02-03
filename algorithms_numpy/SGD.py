import numpy as np
import math
from typing import Tuple, Dict, Any

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
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
        lr_exp: float = 0.67,  # Do not use 1 for averaged algorithms, nor for SGD since we dont know the minimal constant
        lr_const: float = 1.0,
        lr_add_iter: int = 0,
        averaged: bool = False,
        log_weight: float = 2.0,
    ):
        # Name for plotting
        self.name = (
            ("W" if averaged and log_weight != 0.0 else "")
            + ("A" if averaged else "")
            + "SGD"
            + (f" α={lr_exp}")
            + (f" c_α={lr_const}" if lr_const != 1.0 else "")
        )
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter

        super().__init__(param, objective_function, averaged, log_weight)

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
        grad = self.objective_function.grad(data, self.param_not_averaged)

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        self.param_not_averaged -= learning_rate * grad

        if self.averaged:
            self.update_averaged_param()
