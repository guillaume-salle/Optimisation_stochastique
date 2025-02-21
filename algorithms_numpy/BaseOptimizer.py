import numpy as np
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

from objective_functions_numpy_online import BaseObjectiveFunction


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
    DEFAULT_LR_ADD_ITER = 10
    DEFAULT_MULTIPLY_EXP = (
        0.33  # multiply the learning rate constant by an exponent of the dimension
    )
    DEFAULT_LOG_WEIGHT = 2.0

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int | None,
        batch_size_power: int | None,
        averaged: bool | None,
        lr_exp: float | None,
        lr_const: float | None,
        lr_add_iter: int | None,
        log_weight: float | None,
        multiply_lr_const: bool | None,
        multiply_exp: float | None,
    ) -> None:
        """
        Initialize the optimizer with parameters.
        Also initializes a non-averaged parameter, copy of the initial parameter if averaged.

        Args:
            param (np.ndarray): The initial parameters for the optimizer.
            obj_function (BaseObjectiveFunction): The objective function to optimize.
            batch_size (int): The batch size for optimization.
            batch_size_power (int): The power of the dimension for the batch size.
            lr_exp (float): The exponent for the learning rate.
            lr_const (float): The constant for the learning rate.
            lr_add_iter (int): The number of iterations to add to the learning rate.
            averaged (bool): Whether to use an averaged parameter
            log_exp (float): Exponent for the logarithmic weight.
            multiply_lr_const (bool): Whether to multiply the learning rate constant by an exponent of the dimension.
            multiply_exp (float): The exponent for the multiplication of the learning rate constant.
        """
        self.param = param
        self.obj_function = obj_function
        # Batch size is either given or if not, calculated from the power of the dimension
        if batch_size is not None:
            self.batch_size = batch_size
            self.batch_size_power = np.log(batch_size) / np.log(param.shape[0])
        else:
            self.batch_size_power = batch_size_power if batch_size_power is not None else 0
            self.batch_size = param.shape[0] ** self.batch_size_power
        # Not averaged by default
        self.averaged = averaged if averaged is not None else False
        # Learning rate exponent set to 1 for non averaged algorithms, otherwise set to the default value
        if lr_exp is not None:
            self.lr_exp = lr_exp
        else:
            self.lr_exp = 1.0 if not averaged else self.DEFAULT_LR_EXP
        self.lr_const = lr_const if lr_const is not None else self.DEFAULT_LR_CONST
        # Multiply the learning rate constant by an exponent of the dimension
        if multiply_lr_const is not None and multiply_lr_const:
            if multiply_exp is None:
                multiply_exp = self.DEFAULT_MULTIPLY_EXP
            self.lr_const *= param.shape[0] ** multiply_exp
        self.lr_add_iter = lr_add_iter if lr_add_iter is not None else self.DEFAULT_LR_ADD_ITER
        self.log_weight = log_weight if log_weight is not None else self.DEFAULT_LOG_WEIGHT

        self.name = (
            ("S" if self.batch_size_power != 0 else "")  # S for Streaming
            + ("W" if averaged and log_weight != self.DEFAULT_LOG_WEIGHT else "")  # W for Weighted
            + ("A" if averaged else "")  # A for Averaged
            + self.name
            + (f" α={lr_exp}")
            + (f" c_α={lr_const}" if lr_const != self.DEFAULT_LR_CONST else "")
            + (f" c_α*dim^{multiply_exp}" if multiply_lr_const else "")
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
