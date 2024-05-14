from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    @abstractmethod
    def __call__(
        self,
        data: Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        h: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_theta_dim(
        self, data: Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def grad(
        self,
        data: Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        h: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(
        self,
        data: Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def grad_and_riccati(
        self,
        data: Tuple[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        h: torch.Tensor,
        iter: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Riccati is not implemented for this objective function"
        )
