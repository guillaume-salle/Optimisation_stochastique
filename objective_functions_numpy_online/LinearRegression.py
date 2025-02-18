import numpy as np
from typing import Tuple

from objective_functions_numpy_online import (
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
        Compute the linear regression loss, works with a mini_batch or a single data point
        """
        X, y = data
        if self.bias:
            if X.ndim == 1:
                X = add_bias_1d(X)
            else:
                X = add_bias(X)
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
        Compute the gradient of the linear regression loss, works only for a single data point
        """
        X, y = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        Y_pred = np.dot(X, param)
        grad = (Y_pred - y) * X
        return grad

    def hessian(self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the linear regression loss, works only for a single data point
        """
        X, _ = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        return np.outer(X, X)

    def hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, col: int
    ) -> np.ndarray:
        """
        Compute a single column of the Hessian of the linear regression loss,
        works only for a single data poing
        """
        X, _ = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        hessian_col = X[col] * X
        return hessian_col

    def grad_and_hessian(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Hessian of the linear regression loss, works only for a single data point
        """
        X, y = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        Y_pred = np.dot(X, param)
        grad = (Y_pred - y) * X
        hessian = np.outer(X, X)
        return grad, hessian

    def grad_and_hessian_column(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, col: int
    ) -> np.ndarray:
        """
        Compute the gradient and a single culomn of the Hessian of the linear regression loss,
        works only for a single data point
        """
        X, y = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        Y_pred = np.dot(X, param)
        grad = (Y_pred - y) * X
        hessian_col = X[col] * X
        return grad, hessian_col

    def sherman_morrison(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and the Sherman-Morrison term of the linear regression loss,
        works only for a batch_size=1
        """
        X, _ = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        sherman_morrison = X
        return sherman_morrison

    def grad_and_sherman_morrison(
        self, data: Tuple[np.ndarray, np.ndarray], param: np.ndarray, n_iter: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradient and the Sherman-Morrison term of the linear regression loss,
        works only for a batch_size = 1
        """
        X, y = data
        X = X.squeeze()
        if self.bias:
            X = add_bias_1d(X)
        Y_pred = np.dot(X, param)
        grad = (Y_pred - y) * X
        sherman_morrison = X
        return grad, sherman_morrison
