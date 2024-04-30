import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer
    Uses a learning rate lr = c_mu * iteration^(-mu)
    """

    def __init__(
        self, mu: float, c_mu: float = 1.0, thau: float = 1.0, add_iter_lr: int = 20
    ):
        if thau == 0:
            self.name = f"ASGD mu={mu}"
        elif thau == 1.0:
            self.name = f"WASGD mu={mu}"
        else:
            self.name = f"WASGD mu={mu} thau={thau}"
        self.mu = mu
        self.c_mu = c_mu
        self.thau = thau
        self.add_iter_lr = add_iter_lr  # Dont start at 0 to avoid large learning rates at the beginning

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the optimizer state
        """
        self.iter = 0
        self.theta = np.copy(initial_theta)  # Theta not averaged
        self.sum_avg = 0

    def step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        theta_estimate: np.ndarray,  # This is the averaged theta, the not averaged is self.theta
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad = g.grad(X, Y, self.theta)
        learning_rate = self.c_mu * ((self.iter + self.add_iter_lr) ** (-self.mu))
        self.theta += -learning_rate * grad

        avg_coeff = np.log(self.iter + 1) ** self.thau
        self.sum_avg += avg_coeff
        theta_estimate += (self.theta - theta_estimate) * avg_coeff / self.sum_avg
