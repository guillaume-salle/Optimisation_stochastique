from sklearn.datasets import fetch_covtype
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import TensorDataset, random_split


def covtype(test_size: float = 0.2) -> Tuple[TensorDataset, TensorDataset, str]:
    """
    Load the covtype dataset and split it into training and testing sets as PyTorch TensorDatasets.

    Args:
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    Returns a tuple containing the training and testing datasets and the name of the dataset.
    """
    name = "covtype"

    # Fetch the dataset from sklearn
    dataset = fetch_covtype()
    X, y = dataset.data, dataset.target
    y_binary = np.where(y == 1, 1, 0)  # Convert to binary classification

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(y_binary, dtype=torch.long)

    # Create a TensorDataset
    full_dataset = TensorDataset(X_tensor, Y_tensor)

    # Split the dataset into training and testing sets
    train_size = 1 - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset, name
