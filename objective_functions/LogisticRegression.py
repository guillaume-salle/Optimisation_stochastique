from typing import Tuple
import torch
from torch.utils.data import Dataset

from objective_functions import BaseObjectiveFunction, add_bias


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class
    """

    def __init__(self, bias: bool = True):
        self.name = "Logistic model"
        self.bias = bias

    def __call__(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        X, y = data
        X = torch.atleast_2d(X)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        return torch.log(1 + torch.exp(dot_product)) - dot_product * y

    def evaluate_accuracy(self, dataset: Dataset, h: torch.Tensor) -> float:
        """
        Compute the accuracy of the model
        """
        X, Y = dataset.X, dataset.Y
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        predictions = (p > 0.5).float()
        accuracy = (predictions == Y).float().mean().item()
        return accuracy

    def get_theta_dim(self, data: Tuple) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        return X.size(-1) + 1 if self.bias else X.size(-1)

    def grad(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the logistic loss, sum over the batch and normalize by the batch size
        """
        X, Y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        grad = torch.einsum("i,ij->j", p - Y, X) / n
        return grad

    def hessian(self, data: Tuple, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian of the logistic loss, sum over the batch and normalize by the batch size
        """
        X, _ = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        hessian = torch.einsum("i,ij,ik->jk", p * (1 - p), X, X) / n
        return hessian

    def grad_and_hessian(
        self, data: Tuple, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Hessian of the logistic loss,
        sum over the batch and normalize by the batch size
        """
        X, Y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        grad = torch.einsum("i,ij->j", p - Y, X) / n
        hessian = torch.einsum("i,ij,ik->jk", p * (1 - p), X, X) / n
        return grad, hessian

    def grad_and_riccati(
        self, data: Tuple, h: torch.Tensor, iter: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Riccati term of the logistic loss,
        works only with a batch of size 1
        """
        X, Y = data
        n = X.size(0)
        if n != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        grad = (p - Y) * X
        riccati = torch.sqrt(p * (1 - p)) * X
        return grad, riccati
