import numpy as np
from typing import Tuple

from algorithms_numpy import BaseOptimizer
from objective_functions_numpy.streaming import BaseObjectiveFunction


class AdaGrad(BaseOptimizer):
    """
    Adagrad optimizer. Uses a learning rate lr = lr_const * (n_iter + lr_add_iter)^(-lr_exp) for optimization.
    Averaged parameter can be calculated with a logarithmic weight, i.e. the weight is
    calculated as log(n_iter+1)^weight_exp.

    Parameters:
    param (np.ndarray): Initial parameters for the optimizer.
    obj_function (BaseObjectiveFunction): Objective function to optimize.
    mini_batch (int): Size of mini-batch.
    mini_batch_power (float): size of mini-batch as a power of the dimension of the parameter to optimize.
    lr_exp (float): Exponent for learning rate decay.
    lr_const (float): Constant multiplier for learning rate.
    lr_add_iter (int): Additional iterations for learning rate calculation.
    averaged (bool): Whether to use an averaged parameter.
    log_exp (float): Exponent for the logarithmic weight.
    epsilon (float): Small constant to avoid singularity problems.
    true_adagrad (bool): Whether to use the true Adagrad update rule, or one with a decreasing learning rate.
    """

    name = "Ada"

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        mini_batch: int = None,
        mini_batch_power: float = 0.0,
        lr_exp: float = None,
        lr_const: float = BaseOptimizer.DEFAULT_LR_CONST,
        lr_add_iter: int = BaseOptimizer.DEFAULT_LR_ADD_ITER,
        averaged: bool = None,
        log_weight: float = BaseOptimizer.DEFAULT_LOG_WEIGHT,
        multiply_lr: float = BaseOptimizer.DEFAULT_MULTIPLY_LR,
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
            mini_batch=mini_batch,
            mini_batch_power=mini_batch_power,
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
        if self.multiply_lr and self.mini_batch > 1:
            learning_rate = min(learning_rate, self.expected_first_lr)
        self.param_not_averaged -= learning_rate * grad / np.sqrt(self.sum_grad_sq)

        if self.averaged:
            self.update_averaged_param()
