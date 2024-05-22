from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    @abstractmethod
    def __call__(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_theta_dim(self, data: np.ndarray | Tuple[np.ndarray, np.ndarray]) -> int:
        raise NotImplementedError

    @abstractmethod
    def grad(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hessian(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def hessian_column(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray, z: int
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian_column(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray, z: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def riccati(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Riccati is not implemented for this objective function"
        )

    def grad_and_riccati(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Riccati is not implemented for this objective function"
        )
