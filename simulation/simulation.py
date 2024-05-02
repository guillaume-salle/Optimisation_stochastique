import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from tqdm.auto import tqdm
from IPython.display import clear_output

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction


class Simulation:
    """
    Simulation class to run optimization experiments with second order methods,
    on a given function g, with computable gradient and hessian.
    """

    def __init__(
        self,
        g: BaseObjectiveFunction,
        optimizer_list: List[BaseOptimizer],
        e: float,
        dataset: List[Tuple[np.ndarray, np.ndarray]] = None,
        generate_dataset: Callable = None,
        true_theta: np.ndarray = None,
        true_hessian_inv: np.ndarray = None,
    ):
        """
        Initialize the experiment
        """
        if dataset is None and generate_dataset is None:
            raise ValueError("dataset or create_dataset should be set")

        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv
        self.g = g
        self.optimizer_list = optimizer_list
        self.dataset = dataset
        self.generate_dataset = generate_dataset
        self.e = e
        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv

    def generate_initial_theta(self):
        """
        Generate a random initial theta
        """
        if self.dataset is None:
            raise ValueError("dataset is not set")
        theta_dim = self.g.get_theta_dim(self.dataset[0][0])
        if self.true_theta is not None and self.true_theta.shape[0] != theta_dim:
            raise ValueError(
                f"true_theta dim ({self.true_theta.shape[0]}) does not match the dim of theta ({theta_dim}) for g"
            )
        if self.e is None:
            raise ValueError("e is not set for generating random theta")
        loc = self.true_theta if self.true_theta is not None else np.zeros(theta_dim)
        self.initial_theta = loc + self.e * np.random.randn(theta_dim)

    def log_estimation_error(self, theta_errors, hessian_inv_errors, optimizer):
        if self.true_theta is not None:
            theta_errors[optimizer.name].append(
                np.dot(self.theta - self.true_theta, self.theta - self.true_theta)
            )
        if self.true_hessian_inv is not None and optimizer.hessian_inv is not None:
            hessian_inv_errors[optimizer.name].append(
                np.linalg.norm(optimizer.hessian_inv - self.true_hessian_inv, ord="fro")
            )

    def run(
        self, pbars: Tuple[tqdm, tqdm] = None, plot: bool = False
    ) -> Tuple[List[float], List[float]]:
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

        # Pbars are given from run_multiple (workaround vscode bug), otherwise initialize them
        if pbars is not None:
            optimizer_pbar, data_pbar = pbars
            optimizer_pbar.reset(total=len(self.optimizer_list))
        else:
            optimizer_pbar = tqdm(
                total=len(self.optimizer_list), desc="Optimizers", position=0
            )
            data_pbar = tqdm(
                total=len(self.dataset), desc="Data", position=1, leave=False
            )

        # Run the experiment for each optimizer
        for optimizer in self.optimizer_list:
            optimizer.reset(self.initial_theta)
            optimizer_pbar.set_description(optimizer.name)
            self.theta = self.initial_theta.copy()

            # Log initial error
            self.log_estimation_error(theta_errors, hessian_inv_errors, optimizer)

            # Online pass on the dataset
            data_pbar.reset(total=len(self.dataset))
            for X, Y in self.dataset:
                optimizer.step(X, Y, self.theta, self.g)
                self.log_estimation_error(theta_errors, hessian_inv_errors, optimizer)
                data_pbar.update(1)
            optimizer_pbar.update(1)

            # Convert errors to numpy arrays
            if theta_errors is not None:
                theta_errors[optimizer.name] = np.array(theta_errors[optimizer.name])
            if hessian_inv_errors is not None:
                hessian_inv_errors[optimizer.name] = np.array(
                    hessian_inv_errors[optimizer.name]
                )

        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        if plot:
            self.plot_all_errors(theta_errors, hessian_inv_errors, 1)

        return theta_errors, hessian_inv_errors

    def run_multiple_datasets(self, N: int = 100, n: int = 10_000):
        """
        Run the experiment multiple times by generating a new dataset and initial theta each time
        """
        if self.true_theta is None or self.generate_dataset is None:
            raise ValueError("true_theta and/or create_dataset are not set")

        # initialize error dicts, length of error arrays is n + 1 for initial error
        self.theta_errors_avg = {
            optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list
        }
        self.hessian_inv_errors_avg = (
            {optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list}
            if self.true_hessian_inv is not None
            else None
        )

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop
        runs_pbar = tqdm(range(N), desc="Runs", position=0, leave=True)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        # Run the experiment multiple times
        for _ in runs_pbar:
            self.dataset = self.generate_dataset(n, self.true_theta)
            self.generate_initial_theta()
            theta_errors, hessian_inv_errors = self.run([optimizer_pbar, data_pbar])

            # Aggregate the errors
            for name, errors in theta_errors.items():
                self.theta_errors_avg[name] += errors
            if self.true_hessian_inv is not None:
                for name, errors in hessian_inv_errors.items():
                    self.hessian_inv_errors_avg[name] += errors

        # Average the errors
        for name, errors in self.theta_errors_avg.items():
            errors /= N
        if self.true_hessian_inv is not None:
            for name, errors in self.hessian_inv_errors_avg.items():
                errors /= N

        data_pbar.close()
        optimizer_pbar.close()
        runs_pbar.close()

        self.plot_all_errors(self.theta_errors_avg, self.hessian_inv_errors_avg, N)

    def plot_errors(self, errors: dict, title: str, ylabel: str, N: int):
        clear_output(
            wait=True
        )  # Clear the tqdm output, because of bug widgets after reopen

        plt.figure(figsize=(10, 6))
        min_error = float("inf")
        max_error = 0
        for name, errors in errors.items():
            plt.plot(errors, label=name)
            min_error = min(min_error, np.min(errors))
            max_error = max(max_error, np.max(errors))
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(bottom=min(min_error, 1e-3))
        plt.ylim(top=min(max_error, 1e5))
        plt.xlabel("Sample size")
        average = f" averaged over {N} run" + ("s" if N > 1 else "")
        plt.ylabel(ylabel + average)
        plt.title(title)
        plt.suptitle(self.g.name)
        plt.legend()
        plt.show()

    def plot_all_errors(self, theta_errors: dict, hessian_inv_errors: dict, N: int):
        """
        Plot the errors of estimated theta and hessian inverse of all optimizers
        """
        if self.true_theta is not None:
            self.plot_errors(
                theta_errors, f"e = {self.e}", r"$\| \theta - \theta^* \|^2$", N
            )
        if self.true_hessian_inv is not None:
            self.plot_errors(
                hessian_inv_errors, f"e = {self.e}", r"$\| H^{-1} - H^{-1*} \|_F$", N
            )

    plt.show()
