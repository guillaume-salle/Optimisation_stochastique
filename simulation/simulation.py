import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Callable
from functools import partial

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
from objective_functions_numpy.streaming import BaseObjectiveFunction


class Simulation:
    """
    Simulation class to run optimization experiments with second order methods,
    on a given objective function, with computable gradient and hessian.
    """

    REFRESH_OUTPUT = False  # Clear the cell output, because of tqdm bug when reopen notebook

    def __init__(
        self,
        obj_function: BaseObjectiveFunction,  # Not instantiated yet
        optimizer_list: List[partial[BaseOptimizer]],
        generate_dataset: Callable = None,  # generate a dataset with a true_param
        dataset: MyDataset = None,
        test_dataset: MyDataset = None,  # for accuracy evaluation
        dataset_name: str = None,
        true_param: np.ndarray | torch.Tensor = None,
        true_matrix: np.ndarray | torch.Tensor = None,
        initial_param: np.ndarray | torch.Tensor = None,
        r_values: List[float] = [1.0, 5.0],  # noise level for initial parameter
        use_torch: bool = False,
        device: str = None,
    ):
        """
        Initialize the experiment
        """
        if generate_dataset is None and dataset is None:
            raise ValueError("Either `generate_dataset` or `dataset` must be provided.")
        if generate_dataset is not None and true_param is None:
            raise ValueError("`true_param` must be provided if `generate_dataset` is set")

        self.true_param = true_param
        self.true_matrix = true_matrix
        self.obj_function = obj_function
        self.optimizer_list = optimizer_list
        self.generate_dataset = generate_dataset
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.true_matrix = true_matrix
        self.initial_param = initial_param
        self.initial_variances = r_values

        if true_param is not None:
            self.param_dim = true_param.shape[0]
        elif dataset is not None:
            self.param_dim = self.obj_function.get_param_dim(next(iter(dataset)))
        else:
            raise ValueError("true_param or dataset must be provided")
        self.check_duplicate_names()

        # Set the device for torch tensors
        self.use_torch = isinstance(true_param, torch.Tensor) or use_torch
        if use_torch:
            self.device = torch.device(device)
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.obj_function.to(self.device)
            if self.true_param is not None:
                self.true_param = torch.as_tensor(self.true_param, device=self.device)
            if self.true_matrix is not None:
                self.true_matrix = torch.as_tensor(self.true_matrix, device=self.device)
            if self.initial_param is not None:
                self.initial_param = torch.as_tensor(self.initial_param, device=self.device)

    def check_duplicate_names(self):
        """
        Check if there are duplicate names in the optimizer list,
        name are used as keys in the results dictionary and should be unique.
        Also, set the names_list for the optimizers.
        """
        raw_list = []
        names_list = []
        for optimizer_class in self.optimizer_list:
            optimizer = optimizer_class(
                obj_function=self.obj_function, param=np.zeros(self.param_dim)
            )
            if optimizer.name in raw_list:
                count = raw_list.count(optimizer.name)
                name = f"{optimizer.name} ({count})"
            else:
                name = optimizer.name
            names_list.append(name)
            raw_list.append(optimizer.name)
        self.names_list = names_list

    def generate_initial_param(self, variance: float = 1.0):
        """
        Generate a random initial parameter around the true parameter with a given noise level.
        """
        # Generate the initial parameter
        loc = self.true_param if self.true_param is not None else np.zeros(self.param_dim)
        normal = np.random.randn(self.param_dim)
        self.initial_param = loc + variance * normal / np.linalg.norm(normal)

        # Convert to torch tensor if needed
        if self.use_torch:
            self.initial_param = torch.as_tensor(self.initial_param, device=self.device)

    def logging_error(
        self,
        param_errors: dict[str, list],
        matrix_errors: dict[str, list],
        optimizer: BaseOptimizer,
        n: int,
    ):
        """
        Log the estimation error of the parameter and matrix with either numpy or torch tensors
        """
        # Log the estimation error of the parameter
        diff = optimizer.param - self.true_param
        if self.use_torch:
            param_error = torch.linalg.vector_norm(diff).item() ** 2
        else:
            param_error = np.linalg.norm(diff) ** 2
        param_errors[optimizer.name].append((n, param_error))

        # Log the estimation error of condition matrix estimation if a true matrix is provided
        if self.true_matrix is not None and hasattr(optimizer, "matrix"):
            diff_matrix = optimizer.matrix - self.true_matrix
            if self.use_torch:
                matrix_error = torch.linalg.matrix_norm(diff_matrix, ord="fro").item() ** 2
            else:
                matrix_error = np.linalg.norm(diff_matrix, ord="fro") ** 2
            matrix_errors[optimizer.name].append((n, matrix_error))

    def run_track_errors(
        self,
        pbars: Tuple[tqdm, tqdm] = None,
    ) -> Tuple[dict, dict]:
        """
        Run the experiment with all optimizers for a given initial parameter and a dataset generated with a true parameter,
        and track the estimation errors of parameter and matrix if a true matrix is provided.
        Args:
            pbars (Tuple[tqdm, tqdm]): Tuple of progress bars for optimizers and data
        Returns:
            Tuple[dict, dict]: Dictionaries of estimation errors for parameter and matrix for all optimizers
        """
        # Initialize the directories for errors, if true values are provided
        if self.true_param is None:
            raise ValueError("true_param is not set")
        param_errors = {name: [] for name in self.names_list}
        if self.true_matrix is not None:
            matrix_errors = {name: [] for name in self.names_list}
        else:
            matrix_errors = None

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop.
        if pbars is not None:
            optimizer_pbar, data_pbar = pbars
            optimizer_pbar.reset(total=len(self.optimizer_list))
        else:
            optimizer_pbar = tqdm(total=len(self.optimizer_list), desc="Optimizers", position=0)
            data_pbar = tqdm(total=len(self.dataset), desc="Data", position=1, leave=False)

        # Run the experiment for each optimizer
        for i, optimizer_class in enumerate(self.optimizer_list):
            data_pbar.reset(total=len(self.dataset))

            # Initialize the optimizer and parameter
            if self.use_torch:
                self.param = self.initial_param.clone().to(self.device)
            else:
                self.param = self.initial_param.copy()
            optimizer = optimizer_class(obj_function=self.obj_function, param=self.param)
            optimizer.name = self.names_list[i]  # for duplicate names
            optimizer_pbar.set_description(optimizer.name)

            # Initialize the data loader
            batch_size = optimizer.batch_size
            if self.use_torch:
                data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            else:
                data_loader = self.dataset.batch_iter(batch_size)

            # Log initial error
            n = 0
            self.logging_error(param_errors, matrix_errors, optimizer, n)

            # Run the optimizer on the dataset
            for data in data_loader:
                optimizer.step(data)
                n += batch_size
                self.logging_error(param_errors, matrix_errors, optimizer, n)
                data_pbar.update(batch_size)

            optimizer_pbar.update(1)

        # Close the progress bars if not in outer loop
        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        return param_errors, matrix_errors

    def run_track_time(
        self,
        pbars: Tuple[tqdm, tqdm] = None,
    ) -> Tuple[dict, dict]:
        """
        Run the experiment for all optimizers and a given initial parameter and dataset, and track
        the execution time and last parameter error if true parameter is provided, or accuracy if test_dataset is provided.
        Args:
            pbars (Tuple[tqdm, tqdm]): Tuple of progress bars for optimizers and data
        Returns:
            Tuple[dict, dict]: Dictionaries of execution times and last paramter error
                for all optimizers
        """
        # Determine the metric to evaluate
        if self.true_param is not None:
            metrics = "last parameter error"
        elif self.test_dataset is not None:
            metrics = "accuracies"
        else:
            raise ValueError("true_param and/or test_dataset are not set")

        execution_times = {}
        metrics_dict = {}

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop.
        if pbars is not None:
            optimizer_pbar, data_pbar = pbars
            optimizer_pbar.reset(total=len(self.optimizer_list))
        else:
            optimizer_pbar = tqdm(total=len(self.optimizer_list), desc="Optimizers", position=0)
            data_pbar = tqdm(total=len(self.dataset), desc="Data", position=1, leave=False)

        # Run the experiment for each optimizer
        for i, optimizer_class in enumerate(self.optimizer_list):
            data_pbar.reset(total=len(self.dataset))

            # Initialize the parameter
            if self.use_torch:
                self.param = self.initial_param.clone().to(self.device)
            else:
                self.param = self.initial_param.copy()
            optimizer = optimizer_class(obj_function=self.obj_function, param=self.param)
            optimizer.name = self.names_list[i]  # for duplicate names
            optimizer_pbar.set_description(optimizer.name)

            # Initialize the data loader
            batch_size = optimizer.batch_size
            if self.use_torch:
                data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            else:
                data_loader = self.dataset.batch_iter(batch_size)

            # Start timing the optimizer
            time_start = time.time()

            # Run the optimizer on the dataset
            for data in data_loader:
                optimizer.step(data)
                data_pbar.update(batch_size)

            time_end = time.time()
            execution_times[optimizer.name] = time_end - time_start
            if metrics == "last parameter error":
                diff = optimizer.param - self.true_param
                if self.use_torch:
                    error = torch.linalg.vector_norm(diff).item() ** 2
                else:
                    error = np.linalg.norm(diff) ** 2
                metrics_dict[optimizer.name] = error
            elif metrics == "accuracies":
                train_acc = self.obj_function.evaluate_accuracy(self.dataset, self.param)
                test_acc = self.obj_function.evaluate_accuracy(self.test_dataset, self.param)
                metrics_dict[optimizer.name] = {
                    "Training Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                }

            optimizer_pbar.update(1)

        # Close the progress bars if not in outer loop
        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        if metrics == "accuracies":
            # Create a DataFrame with execution times and accuracies
            results_df = pd.DataFrame(
                {
                    "Execution Time": execution_times,
                    "Training Accuracy": {
                        k: v["Training Accuracy"] for k, v in metrics_dict.items()
                    },
                    "Test Accuracy": {k: v["Test Accuracy"] for k, v in metrics_dict.items()},
                }
            ).transpose()

            # Style the DataFrame to highlight the best values
            styled_df = results_df.style.apply(
                # self.highlight_best, order="min", axis=1, subset=["Execution Time"]
                self.highlight_best,
                order="min",
                axis=1,
                subset=pd.IndexSlice["Execution Time", :],
            ).apply(
                self.highlight_best,
                order="max",
                axis=1,
                # subset=["Training Accuracy", "Test Accuracy"],
                subset=pd.IndexSlice[["Training Accuracy", "Test Accuracy"], :],
            )

            # Print the styled DataFrame
            display(styled_df)
            self.stored_accuracies = styled_df

        return execution_times, metrics_dict

    # Called by the previous method (run_track_time) to highlight the best values in a DataFrame
    @staticmethod
    def highlight_best(data, order: str):
        """
        Highlight the minimum or maximum in a DataFrame or Series row with bold font
        """
        if order not in ["min", "max"]:
            raise ValueError("order must be 'min' or 'max'")
        attr = "font-weight: bold"
        if data.ndim == 1:  # Series from a DataFrame.apply(axis=1)
            best = data.min() if order == "min" else data.max()
            return [attr if v == best else "" for v in data]
        else:  # DataFrame direct styling
            best = data.min(axis=1) if order == "min" else data.max(axis=1)
            return pd.DataFrame(
                np.where(data == best[:, None], attr, ""),
                index=data.index,
                columns=data.columns,
            )

    def run_multiple_track_errors(self, N: int = 100, n: int = 10_000, eval_time: bool = False):
        """
        Run the experiment multiple times by generating a new dataset and initial paramter each time.
        Track the estimation errors of paramter, and matrix if a true matrix is provided.
        Plot the errors for each value of e in e_values in separate subplots.
        Args:
            N (int): Number of simulations
            n (int): Number of samples in the dataset
        Returns:
            Tuple[dict, dict]: Dictionaries of execution times and last param errors for all optimizers
        """
        if self.generate_dataset is None:
            raise ValueError("generate_dataset must be set")

        # Initialize error dictionaries to hold results for all the initial variances
        all_param_errors_avg = {}
        if self.true_matrix is not None:
            all_matrix_errors_avg = {}

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop
        N_runs_pbar = tqdm(range(N), position=0, leave=False)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        for variance in self.initial_variances:
            # Initialize the error dictionaries for this variance for initial parameter
            param_errors_avg = {name: 0.0 for name in self.names_list}
            if self.true_matrix is not None:
                matrix_errors_avg = {name: 0.0 for name in self.names_list}

            # Loop of N simulations, each with a new dataset and initial paramter
            N_runs_pbar.reset(total=N)
            N_runs_pbar.set_description(f"Runs for r={variance}")
            for _ in range(N):
                self.dataset, self.dataset_name = self.generate_dataset(n, self.true_param)
                self.generate_initial_param(variance=variance)

                # Run the experiment for each optimizer
                param_errors, matrix_errors = self.run_track_errors(
                    pbars=[optimizer_pbar, data_pbar]
                )

                # Average the errors over the runs
                for name, errors in param_errors.items():
                    param_errors_avg[name] += np.array(errors) / N
                if self.true_matrix is not None:
                    for name, errors in matrix_errors.items():
                        matrix_errors_avg[name] += np.array(errors) / N

                N_runs_pbar.update(1)

            all_param_errors_avg[variance] = param_errors_avg
            if self.true_matrix is not None:
                all_matrix_errors_avg[variance] = matrix_errors_avg

        data_pbar.close()
        optimizer_pbar.close()
        N_runs_pbar.close()

        # Plot errors
        if self.REFRESH_OUTPUT:
            # Clear the cell output, because of tqdm bug widgets after reopen notebook
            clear_output(wait=True)
        ylabel = r"$\| \theta - \theta^* \|^2$"
        self.plot_errors(all_param_errors_avg, ylabel, N)
        if self.true_matrix is not None:
            ylabel = r"$\| H^{-1} - H^{-1*} \|_F$"
            self.plot_errors(all_matrix_errors_avg, ylabel, N)

    def run_multiple_track_time(self, N: int = 100, n: int = 10_000):
        """
        Run the experiment multiple times by generating a new dataset and initial parameter each time.
        Track the execution times and last parameter errors for all optimizers.
        The results are stored in lists to make boxplots.
        Args:
            N (int): Number of simulations
            n (int): Number of samples in the dataset
        Returns:
            Tuple[dict, dict]:
        """
        # Initialize error dictionaries to hold results for each e value
        all_execution_times = {}
        all_metrics = {}

        N_runs_pbar = tqdm(range(N), position=0, leave=False)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        for i, variance in enumerate(self.initial_variances):
            # Initialize the error/time dictionaries
            execution_times = {name: [] for name in self.names_list}
            last_param_errors = {name: [] for name in self.names_list}

            N_runs_pbar.reset(total=N)
            N_runs_pbar.set_description(
                f"{i+1}/{len(self.initial_variances)} Runs for r={variance}"
            )

            for _ in range(N):
                self.dataset, self.dataset_name = self.generate_dataset(n, self.true_param)
                self.generate_initial_param(variance=variance)

                times_run, errors_run = self.run_track_time(pbars=[optimizer_pbar, data_pbar])

                # Store the times and errors for each run
                for name, time in times_run.items():
                    execution_times[name].append(time)
                for name, error in errors_run.items():
                    last_param_errors[name].append(error)

                N_runs_pbar.update(1)

            all_execution_times[variance] = execution_times
            all_metrics[variance] = last_param_errors

        data_pbar.close()
        optimizer_pbar.close()
        N_runs_pbar.close()

        self.boxplot_time_and_errors(all_execution_times, all_metrics, N)

    def plot_errors(self, all_errors_avg: dict, ylabel: str, N: int):
        """
        Plot errors for each value of e in separate subplots, each with its own y-axis scale.

        Args:
        all_errors_avg (dict): Dictionary of errors averaged over runs, keyed by e values.
        ylabel (str): Label for the y-axis.
        N (int): Number of simulations each error average is based on.
        """
        num_e_values = len(all_errors_avg)
        fig, axes = plt.subplots(1, num_e_values, figsize=(10 * num_e_values, 6), sharey=False)

        if num_e_values == 1:
            axes = [axes]  # Make sure axes is iterable

        for ax, (e, errors_dict) in zip(axes, all_errors_avg.items()):
            markers = cycle(["3", "x", "+"])

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Iteration")
            ax.set_title(f"r={e}")

            n_max = len(self.dataset)
            steps_interp = np.arange(1, n_max + 2, 1)

            for idx, (name, errors) in enumerate(errors_dict.items()):
                # Markers to distinguish the optimizers
                markevery = (idx / len(errors_dict), 0.2)
                steps = [s + 1 for s, _ in errors]
                values = [v for _, v in errors]
                values_interp = np.interp(steps_interp, steps, values)
                ax.plot(
                    steps_interp,
                    values_interp,
                    label=name,
                    marker=next(markers),
                    ms=12,
                    markevery=markevery,
                )

            initial_error = errors_dict[self.names_list[0]][0][1]
            ax.set_ylim(top=10 * initial_error)
            # ax.legend(loc="lower left")
            ax.legend(loc="best")

        # Adjust layout to provide space for ylabel
        fig.subplots_adjust(left=0.1)
        fig.text(0.0, 0.5, ylabel, va="center", rotation="vertical")

        plt.suptitle(
            f"{self.obj_function.name} model, {self.dataset_name} dataset, dim={self.true_param.shape[0]}, average over {N} run{'s'*(N!=1)}"
        )
        plt.tight_layout(pad=3.0)
        plt.show()

    def boxplot_time_and_errors(
        self, all_execution_times: dict, all_last_param_error: dict, N: int
    ):
        """
        Make boxplots of the execution times and last parameter errors of all optimizers using seaborn
        """
        if self.REFRESH_OUTPUT:
            # Clear the cell output, because of tqdm bug widgets after reopen notebook
            clear_output(wait=True)

        num_subplots = len(all_execution_times) * 2
        fig, axes = plt.subplots(1, num_subplots, figsize=(4 * num_subplots, 6), sharey=False)

        r_values = list(all_execution_times.keys())

        handles = []
        for i, r in enumerate(r_values):
            ax1 = axes[2 * i]
            ax2 = axes[2 * i + 1]

            # Boxplot execution times with assigned colors
            box1 = sns.boxplot(data=all_execution_times[r], ax=ax1)
            ax1.set_title(f"r={r}")
            ax1.set_ylabel("time (s)")
            ax1.set_ylim(bottom=0)
            ax1.set_xticklabels([])  # Remove x-axis labels

            if not handles:
                # Collect handles and labels for the legend from the first set of plots
                for patch, label in zip(box1.patches, all_execution_times[r].keys()):
                    handles.append(mpatches.Patch(color=patch.get_facecolor(), label=label))

            # Detect outliers in errors
            errors_dict = all_last_param_error[r]
            all_errors = np.concatenate(list(errors_dict.values()))
            global_median = np.median(all_errors)
            threshold = 100 * global_median
            normalized_errors = {
                k: v if np.median(v) < threshold else [0] for k, v in errors_dict.items()
            }
            outlier_optimizers = [k for k, v in errors_dict.items() if np.median(v) >= threshold]

            # Boxplot only normal optimizers for errors
            box2 = sns.boxplot(data=list(normalized_errors.values()), ax=ax2)
            ax2.set_title(f"r={r}")
            ax2.set_ylabel("error")
            ax2.set_xticklabels([])  # Remove x-axis labels
            ax2.set_ylim(bottom=0)

            # Print/Annotate excluded optimizers
            if outlier_optimizers:
                print(f"Excluded optimizers (diverged) for r={r}: {outlier_optimizers}")
                ax2.text(
                    0.5,
                    0.9,
                    "⚠️ Some optimizers excluded",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    color="red",
                )

        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)

        plt.suptitle(
            f"{self.obj_function.name} model, {self.dataset_name} dataset, dim={self.true_param.shape[0]}, average over {N} run{'s'*(N!=1)}"
        )
        plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend
        plt.show()
