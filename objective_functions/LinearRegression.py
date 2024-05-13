import torch
from typing import Any, Tuple

from objective_functions import BaseObjectiveFunction, add_bias


class LinearRegression(BaseObjectiveFunction):
    """
    Linear Regression class
    """

    def __init__(self, bias: bool = True):
        self.bias = bias
        self.name = "Linear model"

    def __call__(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X, y = data
        X = torch.atleast_2d(X)
        if self.bias:
            X = add_bias(X)
        Y_pred = torch.einsum("ni,i->n", X, h)
        error = Y_pred - y
        return 0.5 * error**2

    def get_theta_dim(self, data: Tuple) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        return X.size(-1) + 1 if self.bias else X.size(-1)

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
        error = Y_pred - y
        grad = torch.einsum("n,ni->i", error, X) / n
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
        error = Y_pred - y
        grad = torch.einsum("n,ni->i", error, X) / n
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
        return grad, X
