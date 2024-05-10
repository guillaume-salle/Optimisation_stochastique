from typing import Any
from abc import ABC, abstractmethod

from objective_functions import BaseObjectiveFunction


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
        data: Any,
        theta: torch.Tensor,
        g: BaseObjectiveFunction,
    ) -> None:
        """
        Perform one optimization step.
        Should be implemented by subclasses.
        """
        pass
