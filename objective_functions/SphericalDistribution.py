import numpy as np
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction


class SphericalDistribution(BaseObjectiveFunction):
    """
    Spherical distribution objective function
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Spherical Distribution"

    def __call__(self, X: Any, h: np.ndarray) -> np.ndarray:
        a = h[:-1]
        b = h[-1]
        return 0.5 * (np.linalg.norm(X - a) - b) ** 2

    def get_theta_dim(self, X: np.ndarray) -> int:
        """
        Return the dimension of theta
        """
        return X.shape[-1] + 1

    def grad(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        a = h[:-1].copy()
        b = h[-1]
        norm = np.linalg.norm(X - a)
        grad_a = 0 if norm == 0 else a - X + b * (X - a) / norm
        grad_b = b - norm
        return np.append(grad_a, grad_b)

    def hessian(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        a = h[:-1].copy()
        b = h[-1]
        norm = np.linalg.norm(X - a)
        if norm == 0:
            return None
        eye_part = (1 - b / norm) * np.eye(len(a))
        vector_part = (X - a)[:, None] / norm
        transpose_vector_part = vector_part.T * norm

        hessian = np.block(
            [[eye_part, vector_part], [transpose_vector_part, np.array([[1]])]]
        )
        return hessian

    def grad_and_hessian(
        self, X: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = h[:-1].copy()
        b = h[-1]
        norm = np.linalg.norm(X - a)
        if norm == 0:
            return np.array([0, b]), None
        else:
            grad = np.concatenate([a - X + b * (X - a) / norm, np.array([b - norm])])
            eye_part = (1 - b / norm) * np.eye(len(a)) + (b / norm**3) * np.outer(
                X - a, X - a
            )
            vector_part = (X - a)[:, np.newaxis] / norm
            scalar_part = np.array([[1]])
            top_right = vector_part
            bottom_left = vector_part.T

            hessian = np.block([[eye_part, top_right], [bottom_left, scalar_part]])
            return grad, hessian
