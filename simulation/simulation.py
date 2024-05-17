import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Callable

import matplotlib.patches as mpatches

# Imports for visualization
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from IPython.display import clear_output
from itertools import cycle

import torch
from torch.utils.data import Dataset, DataLoader

# My imports
from datasets_numpy import MyDataset
from algorithms_numpy import BaseOptimizer
from objective_functions_numpy_streaming import BaseObjectiveFunction


class Simulation:
    """
    Simulation class to run optimization experiments with second order methods,
    on a given function g, with computable gradient and hessian.
    """

    def __init__(
        self,
        g: BaseObjectiveFunction,
        optimizer_list: List[BaseOptimizer],
        batch_size: int | str,
        dataset: Dataset | MyDataset = None,
        test_dataset: Dataset | MyDataset = None,
        generate_dataset: Callable = None,
        dataset_name: str = None,
        true_theta: np.ndarray | torch.Tensor = None,
        true_hessian_inv: np.ndarray | torch.Tensor = None,
        initial_theta: np.ndarray | torch.Tensor = None,
        e_values: List[float] = [1.0, 2.0],
        use_torch: bool = False,
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
        self.batch_size = batch_size
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.generate_dataset = generate_dataset
        self.dataset_name = dataset_name
        self.true_theta = true_theta
        self.true_hessian_inv = true_hessian_inv
        # If true values are not provided, set the logging function to None
        if true_theta is None:
            self.logging_estimation_error = lambda *args: None
        elif use_torch:
            self.logging_estimation_error = self.logging_estimation_error_torch
        self.initial_theta = initial_theta
        self.e_values = e_values

        # Set the device for torch tensors
        self.use_torch = use_torch
        if use_torch:
            self.device = torch.device(device)
            if device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            self.g.to(self.device)
            if self.true_theta is not None:
                self.true_theta = torch.as_tensor(self.true_theta, device=self.device)
            if self.true_hessian_inv is not None:
                self.true_hessian_inv = torch.as_tensor(
                    self.true_hessian_inv, device=self.device
                )
            if self.initial_theta is not None:
                self.initial_theta = torch.as_tensor(
                    self.initial_theta, device=self.device
                )

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
        loc = self.true_theta if self.true_theta is not None else np.zeros(theta_dim)
        self.initial_theta = loc + e * np.random.randn(theta_dim)

    def logging_estimation_error(
        self,
        theta_errors: dict[str, list],
        hessian_inv_errors: dict[str, list],
        optimizer,
    ):
        """
        Log the estimation error of theta and hessian inverse with numpy arrays
        """
        if self.true_theta is not None:
            diff = self.theta - self.true_theta
            theta_errors[optimizer.name].append(np.dot(diff, diff))
        if self.true_hessian_inv is not None and hasattr(optimizer, "hessian_inv"):
            hessian_inv_errors[optimizer.name].append(
                np.linalg.norm(optimizer.hessian_inv - self.true_hessian_inv, ord="fro")
            )

    def logging_estimation_error_torch(
        self,
        theta_errors: dict[str, list],
        hessian_inv_errors: dict[str, list],
        optimizer,
    ):
        """
        Log the estimation error of theta and hessian inverse with torch tensors
        """
        if self.true_theta is not None:
            diff = self.theta - self.true_theta
            theta_errors[optimizer.name].append(torch.dot(diff, diff).item())
        if self.true_hessian_inv is not None and hasattr(optimizer, "hessian_inv"):
            hessian_inv_errors[optimizer.name].append(
                torch.norm(
                    optimizer.hessian_inv - self.true_hessian_inv, p="fro"
                ).item()
            )

    def run(
        self, pbars: Tuple[tqdm, tqdm] = None, eval_time: bool = False
    ) -> Tuple[dict, dict]:
        """
        Run the experiment for a given initial theta, a dataset and a list of optimizers
        """
        if self.initial_theta is None:
            raise ValueError("initial theta is not set")
        if self.dataset is None:
            raise ValueError("dataset is not set")
        batch_size = (
            len(self.initial_theta)
            if self.batch_size == "streaming"
            else self.batch_size
        )

        # Initialize the directories for errors if true values are provided and eval_time is False
        if self.true_theta is not None and eval_time is False:
            theta_errors = {optimizer.name: [] for optimizer in self.optimizer_list}
        else:
            theta_errors = None
        if self.true_hessian_inv is not None and eval_time is False:
            hessian_inv_errors = {
                optimizer.name: [] for optimizer in self.optimizer_list
            }
        else:
            hessian_inv_errors = None

        # Store accuracies if test dataset is provided
        accuracies = {} if self.test_dataset is not None else None

        # Store execution times if eval_time is True, and last theta error if true_theta is provided
        execution_times = {} if eval_time is True else None
        last_theta_error = (
            {} if eval_time is True and self.true_theta is not None else None
        )

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop.
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
            optimizer_pbar.set_description(optimizer.name)

            # Initialize the theta
            if self.use_torch:
                self.theta = self.initial_theta.clone().to(self.device)
            else:
                self.theta = self.initial_theta.copy()

            # Reset the progress bar for the data and log the initial error
            data_pbar.reset(total=len(self.dataset))
            if eval_time is False:
                self.logging_estimation_error(
                    theta_errors, hessian_inv_errors, optimizer
                )

            # Initialize the data loader
            if self.use_torch:
                data_loader = DataLoader(
                    self.dataset, batch_size=batch_size, shuffle=False
                )
            else:
                data_loader = self.dataset.batch_iter(batch_size)

            # Start timing the optimizer
            time_start = time.time()
            optimizer.reset(self.initial_theta)

            # Run the optimizer on the dataset
            for data in data_loader:
                optimizer.step(data, self.theta, self.g)
                if eval_time is False:
                    self.logging_estimation_error(
                        theta_errors, hessian_inv_errors, optimizer
                    )
                data_pbar.update(batch_size)

            if eval_time:
                time_end = time.time()
                execution_times[optimizer.name] = time_end - time_start
                if self.true_theta is not None:
                    last_theta_error[optimizer.name] = np.dot(
                        self.theta - self.true_theta, self.theta - self.true_theta
                    )

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

        # Close the progress bars
        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        # Display accuracies if test dataset is provided
        if self.test_dataset is not None:
            accuracies_df = pd.DataFrame(accuracies)
            styled_df = accuracies_df.style.apply(self.highlight_max, axis=1)
            if self.dataset is not None:
                title = (
                    f"Accuracy on {self.dataset_name} dataset, batch size={batch_size}"
                )
                styled_df.set_caption(title)

            # Clear the cell output, because of tqdm bug widgets after reopen
            clear_output(wait=True)

            display(styled_df)

        if eval_time:
            return execution_times, last_theta_error
        else:
            return theta_errors, hessian_inv_errors

    def run_multiple_datasets(
        self, N: int = 100, n: int = 10_000, eval_time: bool = False
    ):
        """
        Run the experiment multiple times by generating a new dataset and initial theta each time
        """
        if self.true_theta is None or self.generate_dataset is None:
            raise ValueError("true_theta and/or create_dataset are not set")

        # Initialize error dictionaries to hold results for each e value
        if eval_time:
            all_execution_times = {}
            all_last_theta_error = {}
        else:
            all_theta_errors_avg = {}
            all_hessian_inv_errors_avg = {}

        runs_pbar = tqdm(range(N), position=0, leave=False)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        for e in self.e_values:
            # Initialize the error/time dictionaries
            if eval_time:
                execution_times_list = {
                    optimizer.name: [] for optimizer in self.optimizer_list
                }
                last_theta_errors_list = {
                    optimizer.name: [] for optimizer in self.optimizer_list
                }
            else:
                theta_errors_avg = {
                    optimizer.name: 0.0 for optimizer in self.optimizer_list
                }
                hessian_inv_errors_avg = (
                    {optimizer.name: 0.0 for optimizer in self.optimizer_list}
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
                if (
                    eval_time
                ):  # run with eval_time=True to get execution times and last theta error
                    execution_time, last_theta_error = self.run(
                        pbars=[optimizer_pbar, data_pbar], eval_time=True
                    )
                    for name, time in execution_time.items():
                        execution_times_list[name].append(time)
                    for name, error in last_theta_error.items():
                        last_theta_errors_list[name].append(error)
                else:  # run with eval_time=False to get theta and hessian inverse errors
                    theta_errors, hessian_inv_errors = self.run(
                        pbars=[optimizer_pbar, data_pbar]
                    )
                    for name, errors in theta_errors.items():
                        theta_errors_avg[name] += np.array(errors) / N
                    if self.true_hessian_inv is not None:
                        for name, errors in hessian_inv_errors.items():
                            hessian_inv_errors_avg[name] += np.array(errors) / N

                runs_pbar.update(1)

            if eval_time:
                all_execution_times[e] = execution_times_list
                all_last_theta_error[e] = last_theta_errors_list
            else:
                all_theta_errors_avg[e] = theta_errors_avg
                all_hessian_inv_errors_avg[e] = hessian_inv_errors_avg

        data_pbar.close()
        optimizer_pbar.close()
        runs_pbar.close()

        if eval_time:
            self.plot_time_and_errors(all_execution_times, all_last_theta_error, N)
        else:
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

        # Adjust layout to provide space for ylabel
        fig.subplots_adjust(left=0.1)
        fig.text(0.0, 0.5, ylabel, va="center", rotation="vertical")

        plt.suptitle(
            f"{self.g.name} model, {self.dataset_name} dataset, average over {N} run{'s'*(N!=1)}, batch size={self.batch_size}"
        )
        plt.tight_layout(pad=3.0)
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

    def plot_time_and_errors(
        self, all_execution_times_avg: dict, all_last_theta_error_avg: dict, N: int
    ):
        """
        Plot the execution times and last theta errors of all optimizers using seaborn
        """
        # Clear the cell output, because of tqdm bug widgets after reopen
        clear_output(wait=True)

        num_subplots = len(all_execution_times_avg) * 2
        fig, axes = plt.subplots(
            1, num_subplots, figsize=(4 * num_subplots, 6), sharey=False
        )

        e_values = list(all_execution_times_avg.keys())

        # Initialize lists for legend handles and labels
        handles = []
        labels = []

        for i, e in enumerate(e_values):
            ax1 = axes[2 * i]
            ax2 = axes[2 * i + 1]

            # Boxplot with seaborn
            box1 = sns.boxplot(data=all_execution_times_avg[e], ax=ax1)
            box2 = sns.boxplot(data=all_last_theta_error_avg[e], ax=ax2)

            ax1.set_title(f"e={e}")
            ax1.set_ylabel("time (s)")
            ax1.set_xticklabels([])  # Remove x-axis labels
            ax2.set_ylabel("error")
            ax2.set_xticklabels([])  # Remove x-axis labels

            # Collect handles and labels for the legend from the first set of plots
            if i == 0:
                for patch, label in zip(
                    box1.patches, all_execution_times_avg[e].keys()
                ):
                    handles.append(
                        mpatches.Patch(color=patch.get_facecolor(), label=label)
                    )

        # Create a single legend outside the plotting area
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.05),
        )

        plt.suptitle(
            f"{self.g.name} model, {self.dataset_name} dataset, average over {N} run{'s'*(N!=1)}, batch size={self.batch_size}"
        )
        plt.tight_layout(
            rect=[0, 0.05, 1, 0.95]
        )  # Adjust layout to make room for the legend
        plt.show()
