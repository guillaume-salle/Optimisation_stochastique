import numpy as np
from typing import Tuple

from algorithms_numpy.estim_hessian import BaseInverseEstimator

class algo(BaseInverseEstimator):
    '''
    Estimation of the inverse of a matrix from random samples
    '''

    def __init__(self, gamma: float, c_gamma: float = 1.0, add_iter: int = 0, dim: int):
        self.name = "algo" + f" γ={gamma}"
        self.gamma = gamma
        self.c_gamma = c_gamma
        self.add_iter = add_iter
        self.dim = dim

    def step(
        self,
        random_matrix: np.ndarray,

    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(data, theta_estimate)
        learning_rate = self.c_alpha * ((self.iter + self.add_iter_theta) ** (-self.alpha))
        theta_estimate += -learning_rate * grad
