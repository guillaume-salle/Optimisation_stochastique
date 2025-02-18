from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseObjectiveFunction(ABC):
    """
    Abstract base class for different objective functions
    """

    @abstractmethod
    def __call__(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_param_dim(self, data: np.ndarray | Tuple[np.ndarray, np.ndarray]) -> int:
        raise NotImplementedError

    @abstractmethod
    def grad(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def hessian(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def hessian_column(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray, z: int
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def grad_and_hessian_column(
        self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray, z: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # Methods to be implemented if random hessians are rank 1
    # def sherman_morrison(
    #     self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     raise NotImplementedError("sherman_morrison is not implemented for this objective function")

    # def grad_and_sherman_morrison(
    #     self, data: np.ndarray | Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     raise NotImplementedError("sherman_morrison is not implemented for this objective function")
