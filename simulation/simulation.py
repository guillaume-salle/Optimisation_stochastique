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
        self.initial_radii = r_values

        if true_param is not None:
            self.param_dim = true_param.shape[0]
        elif dataset is not None:
            self.param_dim = self.obj_function.get_param_dim(next(iter(dataset)))
        else:
            raise ValueError("true_param or dataset must be provided")
        self.check_duplicate_names()

        # Set the device for torch tensors
        self.use_torch = isinstance(true_param, torch.Tensor) or use_torch
        # if use_torch:
        #     self.device = torch.device(device)
        #     if device is None:
        #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.obj_function.to(self.device)
        #     if self.true_param is not None:
        #         self.true_param = torch.as_tensor(self.true_param, device=self.device)
        #     if self.true_matrix is not None:
        #         self.true_matrix = torch.as_tensor(self.true_matrix, device=self.device)
        #     if self.initial_param is not None:
        #         self.initial_param = torch.as_tensor(self.initial_param, device=self.device)

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

    def compute_param_error(
        self,
        optimizer: BaseOptimizer,
    ):
        """
        Log the estimation error of the parameter and matrix with either numpy or torch tensors
        """
        # Log the estimation error of the parameter
        diff = optimizer.param - self.true_param
        if self.use_torch:
            error = torch.linalg.vector_norm(diff).item() ** 2
        else:
            error = np.linalg.norm(diff) ** 2
        return error

    def run(
        self,
        pbars: Tuple[tqdm, tqdm] = None,
        metric: str = "last parameter error",
        track_errors: bool = True,
    ) -> Tuple[dict, dict]:
        """
        Args:
            pbars (Tuple[tqdm, tqdm]): Tuple of progress bars for optimizers and data
        Returns:
        """
        if metric not in ["last parameter error", "accuracy"]:
            raise ValueError("metric must be 'last parameter error' or 'accuracy")

        times_dict = {}
        metrics_dict = {}
        if track_errors:
            errors_list = {name: [] for name in self.names_list}

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop.
        if pbars is not None:
            optimizer_pbar, data_pbar = pbars
            optimizer_pbar.reset(total=len(self.optimizer_list))
        else:
            optimizer_pbar = tqdm(total=len(self.optimizer_list), desc="Optimizers", position=0)
            data_pbar = tqdm(total=len(self.dataset), desc="Data", position=1, leave=False)

        # Run the experiment for each optimizer
        for name, optimizer_class in zip(self.names_list, self.optimizer_list):
            data_pbar.reset(total=len(self.dataset))

            # Initialize the parameter and optimizer
            if self.use_torch:
                self.param = self.initial_param.clone().to(self.device)
            else:
                self.param = self.initial_param.copy()
            optimizer = optimizer_class(obj_function=self.obj_function, param=self.param)
            optimizer_pbar.set_description(name)

            # Initialize the data loader
            batch_size = optimizer.batch_size
            if self.use_torch:
                data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            else:
                data_loader = self.dataset.batch_iter(batch_size)

            # Log initial error
            if track_errors:
                n = 0
                errors_list[name].append((n, self.compute_param_error(optimizer)))

            # Start timing the optimizer
            time_start = time.time()

            # Run the optimizer on the dataset
            for data in data_loader:
                optimizer.step(data)
                if track_errors:
                    n += batch_size
                    errors_list[name].append((n, self.compute_param_error(optimizer)))
                data_pbar.update(batch_size)

            time_end = time.time()
            times_dict[name] = time_end - time_start
            if metric == "last parameter error":
                metrics_dict[name] = self.compute_param_error(optimizer)
            elif metric == "accuracy":
                train_acc = self.obj_function.evaluate_accuracy(self.dataset, self.param)
                test_acc = self.obj_function.evaluate_accuracy(self.test_dataset, self.param)
                metrics_dict[name] = {
                    "Training Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                }

            optimizer_pbar.update(1)

        # Close the progress bars if not in outer loop
        if pbars is None:
            optimizer_pbar.close()
            data_pbar.close()

        return times_dict, metrics_dict, errors_list if track_errors else None

    def run_multiple(
        self,
        N: int,
        n: int,
        track_errors: bool = True,
    ):
        """
        Run the experiment multiple times by generating a new dataset and initial paramter each time.
        Track the estimation errors of paramter over iterations and plot.
        Measure the execution times and last parameter errors for all optimizers.
        Args:
            N (int): Number of simulations
            n (int): Number of samples in the dataset
            track_errors (bool): Track the errors of the parameter over the iterations, for simulated data
        Returns:
            Tuple[dict, dict]: Dictionaries of execution times and last param errors for all optimizers
        """
        # Initialize dictionaries to hold results for all the initial variances
        all_times = {}
        all_metrics = {}
        if track_errors:
            all_errors_list_avg = {}

        # tqdm with VScode bugs, have to initialize the bars outside and reset in the loop
        N_runs_pbar = tqdm(range(N), position=0, leave=False)
        optimizer_pbar = tqdm(
            total=len(self.optimizer_list), desc="Optimizers", position=1, leave=False
        )
        data_pbar = tqdm(total=n, desc="Data", position=2, leave=False)

        for radius in self.initial_radii:
            # Initialize the dictionaries for this radius
            all_times[radius] = {name: [] for name in self.names_list}
            all_metrics[radius] = {name: [] for name in self.names_list}
            if track_errors:
                all_errors_list_avg[radius] = {name: 0.0 for name in self.names_list}

            # Loop of N simulations, each with a new dataset and initial paramter
            N_runs_pbar.reset(total=N)
            N_runs_pbar.set_description(f"Runs for r={radius}")
            for _ in range(N):
                self.dataset, self.dataset_name = self.generate_dataset(n, self.true_param)
                self.generate_initial_param(variance=radius)

                # Run the experiment for each optimizer
                times, metrics, errors_list = self.run(
                    pbars=[optimizer_pbar, data_pbar], track_errors=track_errors
                )

                for name in self.names_list:
                    all_times[radius][name].append(times[name])
                    all_metrics[radius][name].append(metrics[name])
                    if track_errors:  # Average over the runs
                        all_errors_list_avg[radius][name] += np.array(errors_list[name]) / N

                N_runs_pbar.update(1)

        data_pbar.close()
        optimizer_pbar.close()
        N_runs_pbar.close()

        if track_errors:  # Plot errors
            if self.REFRESH_OUTPUT:
                # Clear the cell output, because of tqdm bug widgets after reopen notebook
                clear_output(wait=True)
            ylabel = r"$\| \theta - \theta^* \|^2$"
            self.plot_errors(all_errors_list_avg, ylabel, N)

        self.boxplot_time_and_errors(all_times, all_metrics, N, n)

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
                # Interpolate errors to have the same x values
                steps = [s + 1 for s, _ in errors]
                values = [v for _, v in errors]
                values_interp = np.interp(steps_interp, steps, values)

                base_indices = np.geomspace(10, len(steps_interp) / 10, 3)
                log_offset = idx / len(errors_dict)
                offset_indices = base_indices * (1 + log_offset)
                markevery = np.unique(offset_indices.astype(int))

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

    def boxplot_time_and_errors(self, all_times: dict, all_last_error: dict, N: int, n: int):
        """
        Make boxplots of the execution times and last parameter errors of all optimizers using seaborn
        """
        if self.REFRESH_OUTPUT:
            # Clear the cell output, because of tqdm bug widgets after reopen notebook
            clear_output(wait=True)

        num_subplots = len(all_times) * 2  # 2 plots per r value
        fig, axes = plt.subplots(1, num_subplots, figsize=(4 * num_subplots, 6), sharey=False)
        r_values = list(all_times.keys())
        handles = []
        for i, r in enumerate(r_values):
            ax1 = axes[2 * i]
            ax2 = axes[2 * i + 1]

            # Boxplot execution times with assigned colors
            box1 = sns.boxplot(data=all_times[r], ax=ax1)
            ax1.set_title(f"r={r}")
            ax1.set_ylabel("time (s)")
            ax1.set_ylim(bottom=0)
            ax1.set_xticklabels([])  # Remove x-axis labels

            if not handles:
                # Collect handles and labels for the legend from the first set of plots
                for patch, label in zip(box1.patches, all_times[r].keys()):
                    handles.append(mpatches.Patch(color=patch.get_facecolor(), label=label))

            # Detect NaN and outliers in errors
            errors_dict = all_last_error[r]
            filtered_errors = {}
            excluded_inf_nan = []
            for k, v in errors_dict.items():
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    excluded_inf_nan.append(k)
                else:
                    filtered_errors[k] = v

            if filtered_errors:
                all_errors = np.concatenate(list(filtered_errors.values()))
                global_median = np.median(all_errors)
                threshold = 100 * global_median
                normalized_errors = {
                    k: v if np.median(v) < threshold else [] for k, v in filtered_errors.items()
                }
                outlier_optimizers = [
                    k for k, v in filtered_errors.items() if np.median(v) >= threshold
                ]

                box2 = sns.boxplot(data=normalized_errors, ax=ax2)
            else:
                box2 = ax2
                outlier_optimizers = []

            # # Boxplot only normal optimizers for errors
            # box2 = sns.boxplot(data=list(normalized_errors.values()), ax=ax2)

            ax2.set_title(f"r={r}")
            ax2.set_ylabel("error")
            ax2.set_yscale("log")
            ax2.set_xticklabels([])  # Remove x-axis labels

            if excluded_inf_nan:
                print(f"Excluded optimizers (inf/NaN) for r={r}: {excluded_inf_nan}")
                ax2.text(
                    0.5,
                    0.95,
                    "⚠️ inf/NaN:" + ", ".join([name.split()[0] for name in excluded_inf_nan]),
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    color="red",
                )

            # Print/Annotate excluded optimizers
            if outlier_optimizers:
                print(f"Excluded optimizers (diverged) for r={r}: {outlier_optimizers}")
                ax2.text(
                    0.5,
                    0.9,
                    "⚠️ outliers:" + ", ".join([name.split()[0] for name in outlier_optimizers]),
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    color="red",
                )

            if outlier_optimizers:
                print(f"Excluded optimizers (diverged) for r={r}: {outlier_optimizers}")
                text_pos = 0.85 if excluded_inf_nan else 0.95
                ax2.text(
                    0.5,
                    text_pos,
                    "⚠️ Some optimizers excluded (diverged)",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                    color="red",
                )

        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)

        plt.suptitle(
            f"{self.obj_function.name} model, {self.dataset_name} dataset, dim={self.true_param.shape[0]}, n=1e{int(np.log10(n))}, average over {N} run{'s'*(N!=1)}"
        )
        plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend
        plt.show()

    def display_accuracies(self, times: dict, accuracies: dict):
        """
        Display the accuracies DataFrame with highlighted best values
        """
        # Create a DataFrame with execution times and accuracies
        results_df = pd.DataFrame(
            {
                "Execution Time": times,
                "Training Accuracy": {k: v["Training Accuracy"] for k, v in accuracies.items()},
                "Test Accuracy": {k: v["Test Accuracy"] for k, v in accuracies.items()},
            }
        ).transpose()

        # define a function that highlight the best values in a DataFrame
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

        # Style the DataFrame to highlight the best values
        styled_df = results_df.style.apply(
            highlight_best,
            order="min",
            axis=1,
            subset=pd.IndexSlice["Execution Time", :],
        ).apply(
            highlight_best,
            order="max",
            axis=1,
            subset=pd.IndexSlice[["Training Accuracy", "Test Accuracy"], :],
        )

        display(styled_df)

        return styled_df
