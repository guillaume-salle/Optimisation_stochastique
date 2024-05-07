import torch
from torch.utils.data import TensorDataset


def generate_linear_regression(
    n: int, true_theta: torch.Tensor, bias: bool = True
) -> TensorDataset:
    d = len(true_theta)
    if bias:
        X = torch.randn(n, d - 1)
        phi = torch.cat([torch.ones(n, 1), X], dim=1)
    else:
        X = torch.randn(n, d)
        phi = X

    Y = phi @ true_theta + torch.randn(n)

    return TensorDataset(X, Y)
