from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_theta_dim(self, X: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def grad(self, X: np.ndarray, Y: np.ndarray, h: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(
        self, X: np.ndarray, Y: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
