import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from tqdm.auto import tqdm
from IPython.display import clear_output
from itertools import cycle

from optimization_algorithms import BaseOptimizer
from objective_functions import BaseObjectiveFunction
from experiment_datasets import Dataset


class Simulation:
    """
    Simulation class to run optimization experiments with second order methods,
    on a given function g, with computable gradient and hessian.
    """

    def __init__(
        self,
        g: BaseObjectiveFunction,
        optimizer_list: List[BaseOptimizer],
        dataset: Dataset = None,
        test_dataset: Dataset = None,
        generate_dataset: Callable = None,
        true_theta: np.ndarray = None,
        true_hessian_inv: np.ndarray = None,
        initial_theta: np.ndarray = None,
        e_values: List[float] = [1.0, 2.0],
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
        self.test_dataset = test_dataset
        self.generate_dataset = generate_dataset
        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv
        self.initial_theta = initial_theta
        self.e_values = e_values

    def generate_initial_theta(self, e: float = 1.0):
        """
        Generate a random initial theta
        """
        if self.dataset is None:
            raise ValueError("dataset is not set")
        theta_dim = self.g.get_theta_dim(X=next(iter(self.dataset)))
        if self.true_theta is not None and self.true_theta.shape[0] != theta_dim:
            raise ValueError(
                f"true_theta dim ({self.true_theta.shape[0]}) does not match the dim of theta ({theta_dim}) for g"
            )
        loc = self.true_theta if self.true_theta is not None else np.zeros(theta_dim)
        self.initial_theta = loc + e * np.random.randn(theta_dim)

    def log_estimation_error(
        self, theta_errors: dict, hessian_inv_errors: dict, optimizer
    ):
        if self.true_theta is not None:
            theta_errors[optimizer.name].append(
                np.dot(self.theta - self.true_theta, self.theta - self.true_theta)
            )
        if self.true_hessian_inv is not None and optimizer.hessian_inv is not None:
            hessian_inv_errors[optimizer.name].append(
                np.linalg.norm(optimizer.hessian_inv - self.true_hessian_inv, ord="fro")
            )

    def run(self, pbars: Tuple[tqdm, tqdm] = None) -> Tuple[List[float], List[float]]:
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
        accuracies = {} if self.test_dataset is not None else None

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop
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
            for data in self.dataset:
                optimizer.step(data, self.theta, self.g)
                self.log_estimation_error(theta_errors, hessian_inv_errors, optimizer)
                data_pbar.update(1)
            optimizer_pbar.update(1)

            # Calculate and store accuracies if test dataset is provided
            if self.test_dataset is not None:
                train_acc = self.g.evaluate_accuracy(self.dataset, self.theta)
                test_acc = self.g.evaluate_accuracy(self.test_dataset, self.theta)
                accuracies[optimizer.name] = {
                    "Training Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                }

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

        if accuracies is not None:
            accuracies_df = pd.DataFrame(accuracies)
            styled_df = accuracies_df.style.apply(self.highlight_max, axis=1)
            display(styled_df)

        return theta_errors, hessian_inv_errors

    def run_multiple_datasets(self, N: int = 100, n: int = 10_000, save=False):
        """
        Run the experiment multiple times by generating a new dataset and initial theta each time
        """
        if self.true_theta is None or self.generate_dataset is None:
            raise ValueError("true_theta and/or create_dataset are not set")

        # Initialize error dictionaries to hold results for each e value
        all_theta_errors_avg = {}
        all_hessian_inv_errors_avg = {}

        runs_pbar = tqdm(range(N), position=0, leave=False)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        for e in self.e_values:
            self.theta_errors_avg = {
                optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list
            }
            self.hessian_inv_errors_avg = (
                {optimizer.name: np.zeros(n + 1) for optimizer in self.optimizer_list}
                if self.true_hessian_inv is not None
                else None
            )

            runs_pbar.reset(total=N)
            runs_pbar.set_description(f"Runs for e={e}")

            for _ in range(N):
                self.dataset = self.generate_dataset(n, self.true_theta)
                self.generate_initial_theta(e=e)
                theta_errors, hessian_inv_errors = self.run([optimizer_pbar, data_pbar])

                for name, errors in theta_errors.items():
                    self.theta_errors_avg[name] += errors
                if self.true_hessian_inv is not None:
                    for name, errors in hessian_inv_errors.items():
                        self.hessian_inv_errors_avg[name] += errors
                runs_pbar.update(1)

            for name in self.theta_errors_avg:
                self.theta_errors_avg[name] /= N
            if self.true_hessian_inv is not None:
                for name in self.hessian_inv_errors_avg:
                    self.hessian_inv_errors_avg[name] /= N

            all_theta_errors_avg[e] = self.theta_errors_avg
            all_hessian_inv_errors_avg[e] = self.hessian_inv_errors_avg

        data_pbar.close()
        optimizer_pbar.close()
        runs_pbar.close()

        self.plot_all_errors(all_theta_errors_avg, all_hessian_inv_errors_avg, N)

    def plot_errors(self, all_errors_avg: dict, error_type: str, ylabel: str, N: int):
        """
        Plot errors for each value of e in separate subplots, each with its own y-axis scale.

        Args:
        all_errors_avg (dict): Dictionary of errors averaged over runs, keyed by e values.
        error_type (str): Description of the error type for titling.
        ylabel (str): Label for the y-axis.
        N (int): Number of simulations each error average is based on.
        """
        num_e_values = len(all_errors_avg)
        fig, axes = plt.subplots(
            1, num_e_values, figsize=(10 * num_e_values, 6), sharey=False
        )

        if num_e_values == 1:
            axes = [axes]  # Make sure axes is iterable

        for ax, (e, errors_dict) in zip(axes, all_errors_avg.items()):
            markers_cycle = cycle(
                ["3", "x", "+"]
            )  # Different markers for different optimizers
            markevery_cycle = cycle(
                [
                    int(i * len(self.dataset) / 100 + len(self.dataset) / 11)
                    for i in range(6)
                ]
            )

            max_error = 0

            for (name, errors), mk, me in zip(
                errors_dict.items(), markers_cycle, markevery_cycle
            ):
                ax.plot(errors, label=name, marker=mk, markersize=10, markevery=me)
                max_error = max(max_error, np.max(errors))

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(
                top=min(max_error, 1e5)
            )  # Optionally set a sensible upper limit
            ax.set_xlabel("Sample size")
            average = f" averaged over {N} run" + ("s" if N > 1 else "")
            ax.set_ylabel(ylabel + average)
            ax.set_title(f"{error_type} e={e}")
            ax.legend(loc="lower left")

        plt.suptitle(self.g.name)
        plt.tight_layout()
        plt.show()

    def plot_all_errors(
        self, all_theta_errors: dict, all_hessian_inv_errors: dict, N: int
    ):
        """
        Plot the errors of estimated theta and hessian inverse of all optimizers
        """
        # Clear the cell output, because of tqdm bug widgets after reopen
        clear_output(wait=True)

        if self.true_theta is not None:
            self.plot_errors(all_theta_errors, "", r"$\| \theta - \theta^* \|^2$", N)
        if self.true_hessian_inv is not None:
            self.plot_errors(
                all_hessian_inv_errors, "", r"$\| H^{-1} - H^{-1*} \|_F$", N
            )

    @staticmethod
    def highlight_max(data):
        """
        Highlight the maximum in a DataFrame or Series row with bold font
        """
        attr = "font-weight: bold"
        if data.ndim == 1:  # Series from a DataFrame.apply(axis=1)
            is_max = data == data.max()
            return [attr if v else "" for v in is_max]
        else:  # DataFrame direct styling
            return pd.DataFrame(
                np.where(data == data.max(axis=1)[:, None], attr, ""),
                index=data.index,
                columns=data.columns,
            )
