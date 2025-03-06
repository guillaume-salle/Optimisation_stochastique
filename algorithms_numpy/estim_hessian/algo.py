import numpy as np
from typing import Tuple

from objective_functions_numpy.streaming import BaseObjectiveFunction
from algorithms_numpy.estim_hessian import BaseInverseEstimator


class algo(BaseInverseEstimator):
    """
    Estimation of the inverse of a matrix from random samples
    """

    def __init__(
        self,
        matrix: np.ndarray,
        obj_function: BaseObjectiveFunction,
        lr_exp: float,
        lr_const: float = 1.0,
        lr_add_iter: int = 0,
    ):
        self.name = "algo" + f" γ={lr_exp}"
        self.gamma = lr_exp
        self.c_gamma = lr_const
        self.add_iter = lr_add_iter
        self.dim = dim

    def step(
        self,
        data: np.ndarray | Tuple[np.ndarray, np.ndarray],
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(data, theta_estimate)
        learning_rate = self.c_alpha * ((self.iter + self.add_iter_theta) ** (-self.alpha))
        theta_estimate += -learning_rate * grad
