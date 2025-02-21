import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_online import BaseObjectiveFunction


class AdaGrad(BaseOptimizer):
    """
    Adagrad optimizer. Uses a learning rate lr = lr_const * (n_iter + lr_add_iter)^(-lr_exp) for optimization.
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
    log_exp (float): Exponent for the logarithmic weight.
    epsilon (float): Small constant to avoid singularity problems.
    """

    name = "Ada"

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int = None,
        batch_size_power: int = None,
        lr_exp: float = None,
        lr_const: float = None,
        lr_add_iter: int = None,
        averaged: bool = None,
        log_weight: float = None,
        multiply_lr_const: bool = False,
        multiply_exp: float = None,
        epsilon: float = 1e-8,
    ):
        self.epsilon = epsilon

        # Initialize the sum of the gradients
        self.sum_grad_sq = np.ones_like(param)

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

        # Update the sum of the gradients
        self.sum_grad_sq += grad**2

        # Update the non averaged parameter, add the division of sum_grad_sq by n_iter here, hence the +0.5
        learning_rate = self.lr_const * (self.n_iter + self.lr_add_iter) ** (-self.lr_exp + 0.5)
        # learning_rate = min(learning_rate, 1.0) # TODO decide if we want to clip the learning rate
        self.param_not_averaged -= learning_rate * grad / (np.sqrt(self.sum_grad_sq) + self.epsilon)

        if self.averaged:
            self.update_averaged_param()
