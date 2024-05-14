import numpy as np
from typing import Generator, Tuple, Optional
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        """
        Initialize the dataset.

        Parameters:
        X (np.ndarray): Features of the dataset.
        Y (np.ndarray, optional): Labels of the dataset. If None, the dataset consists only of features.
        """
        self.X = X
        self.Y = Y

    def __iter__(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray] | np.ndarray, None, None]:
        """
        Make the dataset iterable, yielding either (x, y) tuples or just x depending on the presence of Y.
        """
        if self.Y is not None:
            for x, y in zip(self.X, self.Y):
                yield (x, y)
        else:
            for x in self.X:
                yield x

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Return the sample at the given index.

        Parameters:
        idx (int): Index of the sample to return.

        Returns:
        tuple or np.ndarray: If Y is not None, return a tuple (x, y). Otherwise, return x.
        """
        if self.Y is not None:
            return (self.X[idx], self.Y[idx])
        else:
            return self.X[idx]
