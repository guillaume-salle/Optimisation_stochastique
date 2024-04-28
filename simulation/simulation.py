import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

# from tqdm.autonotebook import tqdm
from tqdm import tqdm

from optimization_algorithms import BaseOptimizer


class Simulation:
    """
    Simulation class to run optimization experiments with second order methods,
    on a given function g, with computable gradient and hessian.
    """

    def __init__(
        self,
        g: Callable,
        g_grad: Callable,
        g_grad_and_hessian: Callable,
        optimizer_list: List[BaseOptimizer],
        e: float,
        dataset: List[Tuple[np.ndarray, np.ndarray]] = None,
        generate_dataseet: Callable = None,
        true_theta: np.ndarray = None,
        true_hessian_inv: np.ndarray = None,
        theta_dim: int = None,
    ):
        """
        Initialize the experiment
        """
        if dataset is None and generate_dataseet is None:
            raise ValueError("dataset or create_dataset should be set")
        if true_theta is None and theta_dim is None:
            raise ValueError("dim_theta is not set")

        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv
        self.g = g
        self.g_grad = g_grad
        self.g_grad_and_hessian = g_grad_and_hessian
        self.optimizer_list = optimizer_list
        self.dataset = dataset
        self.generate_dataset = generate_dataseet
        self.e = e
        self.true_theta = true_theta
        self.theta_dim = theta_dim if true_theta is None else true_theta.shape[0]
        self.true_hessian_inv = true_hessian_inv

    def generate_initial_theta(self):
        """
        Generate a random initial theta
        """
        if self.e is None:
            raise ValueError("e is not set for generating random theta")
        loc = (
            self.true_theta if self.true_theta is not None else np.zeros(self.theta_dim)
        )
        self.initial_theta = loc + self.e * np.random.randn(self.theta_dim)

    def log_estimation_error(self, theta_errors, hessian_inv_errors, optimizer):
        if theta_errors is not None:
            theta_errors[optimizer.name].append(
                np.dot(self.theta - self.true_theta, self.theta - self.true_theta)
            )
        if hessian_inv_errors is not None and optimizer.hessian_inv is not None:
            hessian_inv_errors[optimizer.name].append(
                np.linalg.norm(optimizer.hessian_inv - self.true_hessian_inv, ord="fro")
            )

    def run(self, plot: bool = False) -> Tuple[List[float], List[float]]:
        """
        Run the experiment for a given initial theta, a dataset and a list of optimizers
        """
        if self.initial_theta is None:
            raise ValueError("initial theta is not set")
        if self.dataset is None:
            raise ValueError("dataset is not set")

        # Initialize the directories for errors if true values are provided
        theta_errors = (
            {optimizer.name: [] for optimizer in self.optimizer_list}
            if self.true_theta is not None
            else None
        )
        hessian_inv_errors = (
            {optimizer.name: [] for optimizer in self.optimizer_list}
            if self.true_hessian_inv is not None
            else None
        )

        # Run the experiment for each optimizer
        for optimizer in tqdm(
            self.optimizer_list, desc="Optimizers", leave=False, position=1
        ):
            self.theta = self.initial_theta.copy()
            optimizer.reset(self.theta_dim)
            # Log initial error
            self.log_estimation_error(theta_errors, hessian_inv_errors, optimizer)

            # Online pass on the dataset
            for X, Y in tqdm(self.dataset, desc="Data", leave=False, position=2):
                optimizer.step(X, Y, self.theta, self.g_grad, self.g_grad_and_hessian)
                self.log_estimation_error(theta_errors, hessian_inv_errors, optimizer)
            # Convert errors to numpy arrays
            if theta_errors is not None:
                theta_errors[optimizer.name] = np.array(theta_errors[optimizer.name])
            if hessian_inv_errors is not None:
                hessian_inv_errors[optimizer.name] = np.array(
                    hessian_inv_errors[optimizer.name]
                )
        if plot:
            self.plot_errors(theta_errors, hessian_inv_errors)

        return theta_errors, hessian_inv_errors

    def run_multiple(self, num_runs: int = 100, n: int = 10_000):
        """
        Run the experiment multiple times by generating a new dataset and initial theta each time
        """
        if self.true_theta is None or self.generate_dataset is None:
            raise ValueError("true_theta and/or create_dataset are not set")

        # length of error arrays is n + 1 for initial error
        self.theta_errors_avg = (
            {optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list}
            if self.true_theta is not None
            else None
        )
        self.hessian_inv_errors_avg = (
            {optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list}
            if self.true_hessian_inv is not None
            else None
        )

        for i in tqdm(range(num_runs), desc="Runs", position=0):
            self.dataset = self.generate_dataset(n, self.true_theta)
            self.generate_initial_theta()
            theta_errors, hessian_inv_errors = self.run()
            if self.true_theta is not None:
                for name, errors in theta_errors.items():
                    self.theta_errors_avg[name] += errors
            if self.true_hessian_inv is not None:
                for name, errors in hessian_inv_errors.items():
                    self.hessian_inv_errors_avg[name] += errors
        if self.true_theta is not None:
            for name, errors in self.theta_errors_avg.items():
                errors /= num_runs
        if self.true_hessian_inv is not None:
            for name, errors in self.hessian_inv_errors_avg.items():
                errors /= num_runs
        self.plot_errors(self.theta_errors_avg, self.hessian_inv_errors_avg)

    def plot_errors(self, theta_errors: dict, hessian_inv_errors: dict):
        """
        Plot the errors of estimated theta and hessian inverse of all optimizers
        """
        if self.true_theta is not None:
            for name, errors in theta_errors.items():
                plt.plot(errors, label=name)
            plt.xlabel("n")
            plt.ylabel("theta estimation squared error")
            plt.title(f"e = {self.e}")
            plt.legend()
            plt.show()
        if self.true_hessian_inv is not None:
            for name, errors in hessian_inv_errors.items():
                plt.plot(errors, label=name)
            plt.xlabel("n")
            plt.ylabel("hessian inverse estimation error")
            plt.title(f"e = {self.e}")
            plt.legend()
            plt.show()
