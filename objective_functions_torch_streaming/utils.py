import torch


def add_bias(X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Add a bias term to the input data
    """
    return torch.cat([torch.ones(X.size(0), 1, device=device), X], dim=1)


def add_bias_1d(X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Add a bias term to the input data for 1D input
    """
    return torch.cat([torch.ones(1, device=device), X])
