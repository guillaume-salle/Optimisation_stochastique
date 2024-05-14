import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from tqdm.auto import tqdm
from IPython.display import clear_output
from itertools import cycle

import torch
from torch.utils.data import Dataset, DataLoader

from algorithms_torch import BaseOptimizer
from objective_functions_torch_streaming import BaseObjectiveFunction


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
        dataset_name: str = None,
        true_theta: torch.Tensor = None,
        true_hessian_inv: torch.Tensor = None,
        initial_theta: torch.Tensor = None,
        e_values: List[float] = [1.0, 2.0],
        device: str = None,
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
        self.dataset_name = dataset_name
        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv
        # If true values are not provided, set the logging function to None
        if true_theta is None and true_hessian_inv is None:
            self.logging_estimation_error = lambda *args: None
        self.initial_theta = initial_theta
        self.e_values = e_values
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_initial_theta(self, e: float = 1.0):
        """
        Generate a random initial theta
        """
        # Get the dimension of theta from the dataset
        if self.dataset is None:
            raise ValueError("dataset is not set")
        theta_dim = self.g.get_theta_dim(data=next(iter(self.dataset)))
        # Check if the true theta has the same dimension
        if self.true_theta is not None and self.true_theta.shape[0] != theta_dim:
            raise ValueError(
                f"true_theta dim ({self.true_theta.size(0)}) does not match the dim of theta ({theta_dim}) for g"
            )
        # Generate the initial theta
        loc = self.true_theta if self.true_theta is not None else torch.zeros(theta_dim)
        self.initial_theta = loc + e * torch.randn(theta_dim)

    def logging_estimation_error(
        self, theta_errors: dict, hessian_inv_errors: dict, optimizer
    ):
        """
        Log the estimation error of theta and hessian inverse
        """
        if self.true_theta is not None:
            diff = self.theta - self.true_theta
            theta_errors[optimizer.name].append(torch.dot(diff, diff).item())
        if self.true_hessian_inv is not None and optimizer.hessian_inv is not None:
            hessian_inv_errors[optimizer.name].append(
                torch.norm(
                    optimizer.hessian_inv - self.true_hessian_inv, p="fro"
                ).item()
            )

    def run(
        self, batch_size: int, pbars: Tuple[tqdm, tqdm] = None
    ) -> Tuple[dict, dict]:
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
        # Store accuracies if test dataset is provided
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
            self.theta = self.initial_theta.clone().to(self.device)

            # Log initial error
            self.logging_estimation_error(theta_errors, hessian_inv_errors, optimizer)

            # Online pass on the dataset
            data_pbar.reset(total=len(self.dataset))
            data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            for data in data_loader:
                optimizer.step(data, self.theta, self.g)
                self.logging_estimation_error(
                    theta_errors, hessian_inv_errors, optimizer
                )
                data_pbar.update(batch_size)
            optimizer_pbar.update(1)

            # Calculate and store accuracies if test dataset is provided
            if self.test_dataset is not None:
                train_acc = self.g.evaluate_accuracy(self.dataset, self.theta)
                train_acc = round(100 * train_acc, 2)  # percentage

                test_acc = self.g.evaluate_accuracy(self.test_dataset, self.theta)
                test_acc = round(100 * test_acc, 2)

                accuracies[optimizer.name] = {
                    "Training Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                }

            # Convert errors to torch tensors for averaging
            if theta_errors is not None:
                theta_errors[optimizer.name] = torch.tensor(
                    theta_errors[optimizer.name]
                )
            if hessian_inv_errors is not None:
                hessian_inv_errors[optimizer.name] = torch.tensor(
                    hessian_inv_errors[optimizer.name]
                )

        # Close the progress bars
        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        # Display accuracies if test dataset is provided
        if self.test_dataset is not None:
            accuracies_df = pd.DataFrame(accuracies)
            styled_df = accuracies_df.style.apply(self.highlight_max, axis=1)
            if self.dataset is not None:
                styled_df.set_caption("Accuracy on " + self.dataset_name + " dataset")
            display(styled_df)

        return theta_errors, hessian_inv_errors

    def run_multiple_datasets(self, N: int = 100, n: int = 10_000, batch_size: int = 1):
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
                optimizer.name: torch.zeros(n + 1) for optimizer in self.optimizer_list
            }
            self.hessian_inv_errors_avg = (
                {
                    optimizer.name: torch.zeros(n + 1)
                    for optimizer in self.optimizer_list
                }
                if self.true_hessian_inv is not None
                else None
            )

            runs_pbar.reset(total=N)
            runs_pbar.set_description(f"Runs for e={e}")

            for _ in range(N):
                self.dataset, self.dataset_name = self.generate_dataset(
                    n, self.true_theta
                )
                self.generate_initial_theta(e=e)
                theta_errors, hessian_inv_errors = self.run(
                    batch_size, pbars=[optimizer_pbar, data_pbar]
                )

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

    def plot_errors(self, all_errors_avg: dict, ylabel: str, N: int):
        """
        Plot errors for each value of e in separate subplots, each with its own y-axis scale.

        Args:
        all_errors_avg (dict): Dictionary of errors averaged over runs, keyed by e values.
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
            markers = cycle(["3", "x", "+"])
            markevery = cycle(
                [
                    int(i * len(self.dataset) / 100 + len(self.dataset) / 11)
                    for i in range(6)
                ]
            )

            for (name, errors), mk, me in zip(errors_dict.items(), markers, markevery):
                ax.plot(errors, label=name, marker=mk, markersize=10, markevery=me)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Sample size")
            ax.set_title(f"e={e}")
            ax.legend(loc="lower left")

        average = f"Average over {N} run" + ("s" if N > 1 else "")
        fig.text(0.04, 0.5, ylabel + average, va="center", rotation="vertical")

        plt.suptitle(self.g.name + ", " + str(self.dataset_name) + " dataset")
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
            ylabel = r"$\| \theta - \theta^* \|^2$"
            self.plot_errors(all_theta_errors, ylabel, N)
        if self.true_hessian_inv is not None:
            ylabel = r"$\| H^{-1} - H^{-1*} \|_F$"
            self.plot_errors(all_hessian_inv_errors, ylabel, N)

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
