import numpy as np
from typing import List, Tuple

from objective_functions import BaseObjectiveFunction


class GeometricMedian(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Geometric median"

    def __call__(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.linalg.norm(X - theta) - np.linalg.norm(X)

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1]

    def grad(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X - theta)
        if norm == 0:
            return np.zeros_like(theta)
        else:
            return -(X - theta) / norm

    def hessian(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X - theta)
        if norm == 0:
            return np.zeros((theta.shape[0], theta.shape[0]))
        else:
            return (
                np.eye(theta.shape[0]) - np.outer(X - theta, X - theta) / norm**2
            ) / norm

    def grad_and_hessian(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        norm = np.linalg.norm(X - theta)
        if norm == 0:
            return np.zeros_like(theta), np.zeros((theta.shape[0], theta.shape[0]))
        else:
            grad = -(X - theta) / norm
            hessian = (
                np.eye(theta.shape[0]) - np.outer(X - theta, X - theta) / norm**2
            ) / norm
            return grad, hessian

    def grad_and_riccati(
        self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray, iter: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad = self.grad(X, Y, theta)
        Z = np.random.randn(theta.shape[0])
        alpha = 1 / (iter * np.log(iter + 1))
        riccati = self.grad(X, theta + alpha * Z) - grad
        return grad, riccati
