import torch


def add_bias(X: torch.Tensor) -> torch.Tensor:
    """
    Add a bias term to the input data
    """
    return torch.cat(
        [torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype), X], dim=1
    )
