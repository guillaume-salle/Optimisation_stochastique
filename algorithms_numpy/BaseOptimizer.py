import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from objective_functions_numpy.streaming import BaseObjectiveFunction


class BaseOptimizer(ABC):
    """Base class for optimizers.

    This class provides a template for creating optimization algorithms.
    Subclasses should implement the `step` methods to define
    the specific behavior of the optimizer.

    Methods:
        step(data: np.ndarray | Tuple[np.ndarray, np.ndarray],
             objective_function: BaseObjectiveFunction) -> None:
            Perform one optimization step. Should be implemented by subclasses.
    """

    DEFAULT_LR_EXP = 0.67  # Default value for the learning rate exponent for averaged algorithms
    DEFAULT_LR_CONST = 1.0
    DEFAULT_LR_ADD_ITER = 50  # TODO : put dim here
    DEFAULT_LOG_WEIGHT = 2.0
    DEFAULT_MULTIPLY_LR = 0.33  # multiply the learning rate by an exponent of the mini-batch size

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        mini_batch: int | None,
        mini_batch_power: float,
        averaged: bool,
        lr_exp: float | None,
        lr_const: float,
        lr_add_iter: int,
        log_weight: float,
        multiply_lr: float,
    ) -> None:
        """
        Initialize the optimizer with parameters.
        Also initializes a non-averaged parameter, copy of the initial parameter if averaged.

        Args:
            param (np.ndarray): The initial parameters for the optimizer.
            obj_function (BaseObjectiveFunction): The objective function to optimize.
            mini_batch (int): The mini-batch size for optimization. If None, calculated from the power of the dimension.
            mini_batch_power (float): The power of the dimension for the mini-batch size.
            lr_exp (float): The exponent for the learning rate. If None, set to 1 for non averaged algorithms and to the default value for averaged algorithms.
            lr_const (float): The constant for the learning rate.
            lr_add_iter (int): The number of iterations to add to the learning rate.
            averaged (bool): Whether to use an averaged parameter
            log_exp (float): Exponent for the logarithmic weight.
            multiply_lr (float): Multiply the learning rate by mini_batch_size^multiply_lr, for mini-batch.
        """
        self.param = param
        self.obj_function = obj_function
        # Batch size is either given or if not, calculated from the power of the dimension
        if mini_batch is not None:
            self.mini_batch = mini_batch
            self.mini_batch_power = np.log(mini_batch) / np.log(param.shape[0])
        else:
            self.mini_batch_power = mini_batch_power
            self.mini_batch = int(param.shape[0] ** self.mini_batch_power)
        self.averaged = averaged
        # Learning rate exponent set to 1 for non averaged algorithms, otherwise set to a default value
        if lr_exp is not None:
            self.lr_exp = lr_exp
        else:
            self.lr_exp = 1.0 if not averaged else self.DEFAULT_LR_EXP
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter
        self.log_weight = log_weight

        # Multiply the learning rate by an exponent of the mini-batch size
        if multiply_lr > 0 and self.mini_batch > 1:
            self.multiply_lr = True  # TODO : check if this is necessary
            self.expected_first_lr = self.lr_const * (1 + self.lr_add_iter) ** (-self.lr_exp)
            self.lr_const *= self.mini_batch**multiply_lr
        else:
            self.multiply_lr = False

        self.name = (
            ("S" if self.mini_batch_power != 0 else "")  # S for Streaming
            # + ("W" if averaged and log_weight != 0.0 else "")  # W for Weighted
            + ("A" if averaged else "")  # A for Averaged
            + self.name
            + (f" α={self.lr_exp}")
            + (f" c_α={lr_const}" if lr_const != self.DEFAULT_LR_CONST else "")
            + (f" p=d^{self.mini_batch_power}" if self.mini_batch_power != 0 else "")
            + (f" c_α*p^{multiply_lr}" if multiply_lr > 0 and self.mini_batch > 1 else "")
        )

        # Copy the initial parameter if averaged, otherwise use the same
        self.param_not_averaged = np.copy(param) if averaged else param

        self.sum_weights = 0
        self.n_iter = 0

    @abstractmethod
    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Perform one optimization step.
        Should be implemented by subclasses.

        Args:
            data (np.ndarray | Tuple[np.ndarray, np.ndarray]): The input data for the optimization step.
        """
        pass

    def update_averaged_param(self) -> None:
        """
        Update the averaged parameter using the current parameter and the sum of weights.
        """
        if self.log_weight > 0:
            weight = np.log(self.n_iter + 1) ** self.log_weight
        else:
            weight = 1
        self.sum_weights += weight
        self.param += (weight / self.sum_weights) * (self.param_not_averaged - self.param)
