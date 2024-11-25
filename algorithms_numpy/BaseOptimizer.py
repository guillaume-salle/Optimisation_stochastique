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

    def __init__(
        self,
        param: np.ndarray,
        objective_function: BaseObjectiveFunction,
        averaged: bool = False,
        weight_exp: float = 0.0,
    ) -> None:
        """
        Initialize the optimizer with parameters.
        Also initializes a non-averaged parameter, copy of the initial parameter if averaged.

        Args:
            param (np.ndarray): The initial parameters for the optimizer.
            averaged (bool): Whether to use an averaged parameter
            weight_exp (float): Exponent for the logarithmic weight.
        """
        self.param = param
        self.objective_function = objective_function

        self.averaged = averaged
        self.param_not_averaged = np.copy(param) if averaged else param
        self.weight_exp = weight_exp
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
        if not self.averaged:
            return

        if self.weight_exp > 0:
            weight = np.log(self.n_iter + 1) ** self.weight_exp
        else:
            weight = 1
        self.sum_weights += weight
        self.param += (weight / self.sum_weights) * (self.param_not_averaged - self.param)
