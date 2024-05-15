import torch
from typing import Tuple

from objective_functions_torch_streaming import (
    BaseObjectiveFunction,
    add_bias,
    add_bias_1d,
)


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear model"

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X, y = data
        if self.bias:
            if X.ndim == 1:
                X = add_bias_1d(X)
            else:
                X = add_bias(X)
        Y_pred = torch.einsum("ni,i->n", X, h)
        return 0.5 * (Y_pred - y) ** 2

    def get_theta_dim(self, data: Tuple) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        if self.bias:
            return X.size(-1) + 1
        else:
            return X.size(-1)

    def grad(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the linear regression loss,
        sum over the batch and normalize by the batch size
        """
        X, y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        Y_pred = torch.einsum("ni,i->n", X, h)
        grad = torch.einsum("n,ni->i", Y_pred - y, X) / n
        return grad

    def hessian(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the linear regression loss,
        sum over the batch and normalize by the batch size
        """
        X, _ = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        hessian = torch.einsum("ni,nj->ij", X, X) / n
        return hessian

    def grad_and_hessian(
        self, data: Tuple, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Hessian of the linear regression loss,
        sum over the batch and normalize by the batch size
        """
        X, y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        Y_pred = torch.einsum("ni,i->n", X, h)
        grad = torch.einsum("n,ni->i", Y_pred - y, X) / n
        hessian = torch.einsum("nj,nk->jk", X, X) / n
        return grad, hessian

    def grad_and_riccati(
        self, data: Tuple, h: torch.Tensor, iter: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Riccati of the linear regression loss
        works only with a batch of size 1
        """
        X, y = data
        n = X.size(0)
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        if self.bias:
            X = add_bias(X)
        X = X.squeeze()
        Y_pred = torch.dot(X, h)
        grad = (Y_pred - y) * X
        riccati = X
        return grad, riccati
