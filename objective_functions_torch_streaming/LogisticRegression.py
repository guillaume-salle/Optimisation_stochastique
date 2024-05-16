import torch
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple

from objective_functions_torch_streaming import (
    BaseObjectiveFunction,
    add_bias,
    add_bias_1d,
)


def sigmoid(z: float):
    """Stable sigmoid function that avoids overflow, works with a single value"""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        return math.exp(z) / (1 + math.exp(z))


class LogisticRegression(BaseObjectiveFunction):
    """
    Logistic Regression class
    """

    def __init__(self, bias: bool = True):
        self.name = "Logistic model"
        self.bias = bias

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the logistic loss, works with a batch or a single data point
        """
        X, y = data
        if self.bias:
            if X.ndim == 1:
                X = add_bias_1d(X)
            else:
                X = add_bias(X)
        dot_product = torch.matmul(X, h)
        return torch.log(1 + torch.exp(dot_product)) - dot_product * y

    def evaluate_accuracy(
        self, dataset: Dataset, h: torch.Tensor, batch_size=512
    ) -> float:
        """
        Compute the accuracy of the model
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        correct_predictions = 0
        total = 0

        for X, Y in dataloader:
            if self.bias:
                X = add_bias(X)
            dot_product = torch.matmul(X, h)
            p = torch.sigmoid(dot_product)
            predictions = (p > 0.5).int()
            correct_predictions += (predictions == Y.int()).sum().item()
            total += Y.size(0)

        return correct_predictions / total

    def get_theta_dim(self, data: Tuple[torch.Tensor, torch.Tensor]) -> int:
        """
        Return the dimension of theta
        """
        X, _ = data
        if self.bias:
            return X.size(-1) + 1
        else:
            return X.size(-1)

    def grad(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the logistic loss, average over the batch
        """
        X, Y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        grad = torch.matmul(p - Y, X) / n
        return grad

    def hessian(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Hessian of the logistic loss, average over the batch
        """
        X, _ = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.matmul(X, h)
        p = torch.sigmoid(dot_product)
        hessian = torch.einsum("n,ni,nj->ij", p * (1 - p) / n, X, X)
        return hessian

    def grad_and_hessian(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Hessian of the logistic loss, average over the batch
        """
        X, Y = data
        n = X.size(0)
        if self.bias:
            X = add_bias(X)
        dot_product = torch.einsum("ni,i->n", X, h)
        p = torch.sigmoid(dot_product)
        grad = torch.einsum("n,ni->i", p - Y, X) / n
        hessian = torch.einsum("n,ni,nj->ij", p * (1 - p), X, X) / n
        return grad, hessian

    def grad_and_riccati(
        self, data: Tuple[torch.Tensor, torch.Tensor], h: torch.Tensor, iter: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient and the Riccati term of the logistic loss,
        works only with a batch of size 1
        """
        X, Y = data
        if X.size(0) != 1:
            raise ValueError("The Riccati term is only defined for a single data point")
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        dot_product = torch.dot(X, h)
        p = sigmoid(dot_product)
        grad = (p - Y) * X
        riccati = math.sqrt(p * (1 - p)) * X
        return grad, riccati
