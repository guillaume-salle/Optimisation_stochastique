import numpy as np
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
    obj_function (BaseObjectiveFunction): Objective function to optimize.
    batch_size (int): Size of the batch.
    batch_size_power (int): batch size as a power of the dimension of the parameter to optimize.
    lr_exp (float): Exponent for learning rate decay.
    lr_const (float): Constant multiplier for learning rate.
    lr_add_iter (int): Additional iterations for learning rate calculation.
    averaged (bool): Whether to use an averaged parameter.
    weight_exp (float): Exponent for the logarithmic weight.
    """

    name = "SGD"

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int = None,
        batch_size_power: int = 0,
        lr_exp: float = None,
        lr_const: float = 1.0,
        lr_add_iter: int = 0,
        averaged: bool = False,
        log_weight: float = 2.0,
        multiply_lr_const: bool = False,
        multiply_exp: float = None,
    ):
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

        Parameters:
        data (np.ndarray | Tuple[np.ndarray, np.ndarray]): The input data for the optimization step.
        """
        self.n_iter += 1
        grad = self.obj_function.grad(data, self.param_not_averaged)

        # Update the non averaged parameter
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp)
        # learning_rate = min(learning_rate, 1.0) # TODO decide if we want to clip the learning rate
        if learning_rate > 1:
            print(f"Learning rate is larger than 1: {learning_rate}")
            raise ValueError("Learning rate is larger than 1")
        self.param_not_averaged -= learning_rate * grad

        if self.averaged:
            self.update_averaged_param()
