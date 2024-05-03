from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    @abstractmethod
    def __call__(self, X: Any, h: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_theta_dim(self, X: Any) -> int:
        raise NotImplementedError

    @abstractmethod
    def grad(self, X: Any, h: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(self, X: Any, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def grad_and_riccati(
        self, X: Any, h: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Riccati is not implemented for this objective function"
        )
