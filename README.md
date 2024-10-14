# Universal Stochastic Newton Algorithm

## Overview

This repository contains implementations of various optimization algorithms, including the Universal Stochastic Newton Algorithm (USNA) and its variants. The algorithms are implemented using both NumPy and PyTorch frameworks.

## Algorithms

### NumPy Implementations

- **BaseOptimizer**: Base class for all optimizers.
- **SGD**: Stochastic Gradient Descent.
- **SNA**: Stochastic Newton Algorithm.
- **SNARiccati**: Stochastic Newton Algorithm with Riccati updates.
- **USNA**: Universal Stochastic Newton Algorithm.
- **UWASNA**: Universal Weighted Averaged Stochastic Newton Algorithm.
- **WASGD**: Weighted Averaged Stochastic Gradient Descent.
- **WASNA**: Weighted Averaged Stochastic Newton Algorithm.

### PyTorch Implementations

- Implementations of the above algorithms using PyTorch can be found in the `algorithms_torch` directory.

## Datasets

- **datasets_numpy**: Datasets for NumPy-based algorithms.
- **datasets_torch**: Datasets for PyTorch-based algorithms.

## Notebooks

- **Notebooks_numpy_online**: Jupyter notebooks for online learning using NumPy.
- **Notebooks_numpy_streaming**: Jupyter notebooks for streaming data using NumPy.
- **Notebooks_tests**: Test notebooks.

## Objective Functions

- **objective_functions_numpy_online**: Objective functions for online learning using NumPy.
- **objective_functions_numpy_streaming**: Objective functions for streaming data using NumPy.
- **objective_functions_torch_streaming**: Objective functions for streaming data using PyTorch.

## Simulation

- The `simulation` directory contains scripts and tools for running simulations.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- PyTorch
- Jupyter Notebook

### Installation

Clone the repository:

```sh
git clone https://github.com/guillaume-salle/Optimisation_stochastique.git
cd Optimisation_stochastique

Install the required packages:

pip install -r requirements.txt

Running Notebooks
Navigate to the desired notebook directory and start Jupyter Notebook:

Open the notebook you want to run 