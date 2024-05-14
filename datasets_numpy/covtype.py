from sklearn.datasets import fetch_covtype
import numpy as np
from typing import Tuple
from experiment_datasets import MyDataset

from sklearn.model_selection import train_test_split


def covtype() -> Tuple[MyDataset, MyDataset]:
    """
    Load the covtype dataset and split it into training and testing sets.
    """
    dataset = fetch_covtype()
    X, y = dataset.data, dataset.target
    y_binary = np.where(y == 1, 1, 0)

    # Split the data into training and testing sets
    X_train, X_test, Y_train_binary, Y_test_binary = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )
    name = "covtype"
    return MyDataset(X_train, Y_train_binary, name), MyDataset(
        X_test, Y_test_binary, name
    )
