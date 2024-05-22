from sklearn.datasets import fetch_covtype
import numpy as np
from typing import Tuple
from datasets_numpy import MyDataset
from sklearn.model_selection import train_test_split


def covtype(test_size: float = 0.2) -> Tuple[MyDataset, MyDataset, str]:
    """
    Load the covtype dataset and split it into training and testing sets.

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

    # Split the data into training and testing sets
    X_train, X_test, Y_train_binary, Y_test_binary = train_test_split(
        X, y_binary, test_size=test_size, random_state=1
    )

    return MyDataset(X_train, Y_train_binary), MyDataset(X_test, Y_test_binary), name
