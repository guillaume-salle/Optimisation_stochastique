import numpy as np
from typing import Tuple

from objective_functions_numpy.streaming import (
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
        self.name = "Linear"

    def __call__(self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray) -> np.ndarray:
        """
        Compute the linear regression loss, works with a batch or a single data point
        """
        X, y = data
        if self.bias:
            if X.ndim == 1:
                X = np.insert(X, 0, 1)
            else:
                batch_size = X.shape[0]
                X = np.hstack([np.ones((batch_size, 1)), X])
        Y_pred = np.dot(X, param)
        return 0.5 * (Y_pred - y) ** 2

    def get_param_dim(self, data: Tuple[np.ndarray, np.ndarray]) -> int:
        """
        Return the dimension of theta, works with a batch or a single data point
        """
        X, _ = data
        if self.bias:
            return X.shape[-1] + 1
        else:
            return X.shape[-1]

    def grad(self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the linear regression loss, average over the batch
        """
        X, y = data
        batch_size = X.shape[0]
        if self.bias:
            X = np.hstack([np.ones((batch_size, 1)), X])
        Y_pred = np.dot(X, param)
        grad = np.dot(X.T, Y_pred - y) / batch_size
        return grad

    def hessian(self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, average over the batch
        """
        X, _ = data
        batch_size = X.shape[0]
        if self.bias:
            X = np.hstack([np.ones((batch_size, 1)), X])
        hessian = np.einsum("ki,kj->ij", X / batch_size, X)
        return hessian

    def hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, col: int
    ) -> np.ndarray:
        """
        Compute a single column of the hessian of the objective function, average over the batch
        """
        X, _ = data
        batch_size = X.shape[0]
        if self.bias:
            X = np.hstack([np.ones((batch_size, 1)), X])
        hessian_col = np.dot(X.T, X[:, col]) / batch_size
        return hessian_col

    def grad_and_hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and a single column of the hessian of the objective function,
        average over the batch
        """
        X, y = data
        batch_size = X.shape[0]
        if self.bias:
            X = np.hstack([np.ones((batch_size, 1)), X])
        Y_pred = np.dot(X, param)
        grad = np.dot(X.T, Y_pred - y) / batch_size
        hessian_col = np.dot(X.T, X[:, col]) / batch_size
        return grad, hessian_col

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss, average over the batch
        """
        X, y = data
        batch_size = X.shape[0]
        if self.bias:
            X = np.hstack([np.ones((batch_size, 1)), X])
        Y_pred = np.dot(X, param)
        grad = np.dot(X.T, Y_pred - y) / batch_size
        hessian = np.einsum("ki,kj->ij", X / batch_size, X)
        return grad, hessian

    def sherman_morrison(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Sherman-Morrison term of the linear regression loss, works only for a batch of size of 1
        """
        X, _ = data
        batch_size = X.shape[0]
        if batch_size != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")
        X = X.squeeze()
        if self.bias:
            X = np.insert(X, 0, 1)
        sherman_morrison = X
        return sherman_morrison

    def grad_and_sherman_morrison(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Sherman-Morrison term of the linear regression loss,
        works only for a batch of size 1
        """
        X, y = data
        batch_size = X.shape[0]
        if batch_size != 1:
            raise ValueError("The Sherman-Morrison update is only possible for a batch size of 1")
        X = X.squeeze()
        if self.bias:
            X = np.insert(X, 0, 1)
        Y_pred = np.dot(X, param)
        grad = (Y_pred - y) * X
        sherman_morisson = X
        return grad, sherman_morisson
