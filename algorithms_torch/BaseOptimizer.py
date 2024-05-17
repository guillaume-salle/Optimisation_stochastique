import torch
from typing import Tuple
from abc import ABC, abstractmethod

from objective_functions_torch_streaming import BaseObjectiveFunction


class BaseOptimizer(ABC):
    """
    Base class for optimizers.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def reset(self, theta_dim: int) -> None:
        """
        Reset the optimizer state.
        Should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def step(
        self,
        data: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        theta: torch.Tensor,
        g: BaseObjectiveFunction,
    ) -> None:
        """
        Perform one optimization step.
        Should be implemented by subclasses.
        """
        pass
