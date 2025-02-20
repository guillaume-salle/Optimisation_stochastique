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

    DEFAULT_MULTIPLY_EXP = 0.5

    def __init__(
        self,
        param: np.ndarray,
        obj_function: BaseObjectiveFunction,
        batch_size: int | None,
        batch_size_power: int,
        lr_exp: float,
        lr_const: float,
        lr_add_iter: int,
        averaged: bool,
        log_weight: float,
        multiply_lr_const: bool,
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
        self.batch_size_power = batch_size_power
        # Batch size is either given or if not, calculated from the power of the dimension
        if batch_size is not None:
            self.batch_size = batch_size
            self.batch_size_power = np.log(batch_size) / np.log(param.shape[0])
        else:
            self.batch_size = param.shape[0] ** batch_size_power
        self.lr_exp = lr_exp
        self.lr_const = lr_const
        self.lr_add_iter = lr_add_iter
        self.averaged = averaged
        # Copy the initial parameter if averaged, otherwise use the same
        self.param_not_averaged = np.copy(param) if averaged else param
        self.log_weight = log_weight
        if multiply_lr_const:
            if multiply_exp is None:
                multiply_exp = self.DEFAULT_MULTIPLY_EXP
            self.lr_const *= param.shape[0] ** multiply_exp

        self.name = (
            ("S" if self.batch_size_power != 0 else "")
            + ("W" if averaged and log_weight != 0.0 else "")
            + ("A" if averaged else "")
            + self.name
            + (f" α={lr_exp}")
            + (f" c_α={lr_const}" if lr_const != 1.0 else "")
            + (f" c_α*dim^{multiply_exp}" if multiply_lr_const else "")
        )

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
