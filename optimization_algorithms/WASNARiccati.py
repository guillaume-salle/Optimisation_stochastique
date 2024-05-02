import numpy as np

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class WASNARiccati(BaseOptimizer):
    """
    Stochastic Newton Algorithm optimizer
    """

    def __init__(
        self,
        nu: float,
        c_nu: float = 1.0,
        tau_theta: float = 2.0,
        add_iter_lr: int = 20,
        lambda_: float = 10.0,  # Weight more the initial identity matrix
    ):
        self.class_name = "WASNARiccati"
        self.name = (
            ("WASNARiccati" if tau_theta != 0.0 else "SNA-Riccati")
            + (f" ν={nu}" if nu != 1.0 else "")
            + (f" τ_theta={tau_theta}" if tau_theta != 2.0 and tau_theta != 0.0 else "")
        )
        self.nu = nu
        self.c_nu = c_nu
        self.tau_theta = tau_theta
        self.add_iter_lr = add_iter_lr
        self.lambda_ = lambda_

    def reset(self, initial_theta: np.ndarray):
        """
        Reset the learning rate and estimate of the hessian
        """
        self.iter = 0
        self.theta_dim = initial_theta.shape[0]
        self.theta_not_avg = np.copy(initial_theta)
        self.sum_weights_theta = 0
        # Weight more the initial identity matrix
        self.hessian_bar_inv = (
            1 / (self.lambda_ * self.theta_dim) * np.eye(self.theta_dim)
        )

    def step(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        g: BaseObjectiveFunction,
    ):
        """
        Perform one optimization step
        """
        self.iter += 1
        grad, phi = g.grad_and_riccati(X, Y, theta, self.iter)

        # Update the hessian estimate
        product = self.hessian_bar_inv @ phi
        denominator = 1 + np.dot(phi, product)
        if np.abs(denominator) < 1e-8:
            print("Denominator too small, update skipped")
        else:
            self.hessian_bar_inv += -np.outer(product, product) / denominator

        # Update the theta estimate
        learning_rate = self.c_nu * (self.iter + self.add_iter_lr) ** (-self.nu)
        self.theta_not_avg += (
            -learning_rate
            * (self.iter + self.lambda_ * self.theta_dim)
            * self.hessian_bar_inv
            @ grad
        )
        weigth_theta = np.log(self.iter + 1) ** self.tau_theta
        self.sum_weights_theta += weigth_theta
        theta += (self.theta_not_avg - theta) * weigth_theta / self.sum_weights_theta
