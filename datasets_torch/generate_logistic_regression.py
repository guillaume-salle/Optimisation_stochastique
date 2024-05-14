from typing import Tuple
import torch
from torch.utils.data import TensorDataset


def generate_logistic_regression(
    n: int, true_theta: torch.Tensor, bias: bool = True
) -> Tuple[TensorDataset, str]:
    """
    Generate data from a logistic regression model for testing.
    """
    name = "logistic regression"

    d = len(true_theta)
    if bias:
        # Create feature matrix X with an additional bias term (column of ones)
        X = torch.randn(n, d - 1)
        phi = torch.cat([torch.ones(n, 1), X], dim=1)
    else:
        X = torch.randn(n, d)
        phi = X

    Y = torch.bernoulli(torch.sigmoid(phi @ true_theta))

    return TensorDataset(X, Y), name
