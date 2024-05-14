import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from objective_functions_numpy_online import BaseObjectiveFunction


class BaseOptimizer(ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def reset(self, theta_dim: int) -> None:
        """
        Reset the optimizer state.
        Should be implemented by subclasses.
        """
        pass

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
        thea_estimate: np.ndarray,
        g: BaseObjectiveFunction,
    ) -> None:
        """
        Perform one optimization step.
        Should be implemented by subclasses.
        """
        pass
