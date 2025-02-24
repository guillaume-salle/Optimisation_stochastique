from abc import abstractmethod
import numpy as np


class BaseInverseEstimator:
    """
    Base class for an estimator of the inverse of a matrix
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def step(self, matrix: np.ndarray) -> None:
        """
        Update the estimator with a new matrix.
        Should be implemented by subclasses.
        """
        pass
