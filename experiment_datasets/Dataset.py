import numpy as np


class Dataset:
    def __init__(self, X, Y=None, name=None):
        """
        Initialize the dataset.

        Parameters:
        X (np.ndarray): Features of the dataset.
        Y (np.ndarray, optional): Labels of the dataset. If None, the dataset consists only of features.
        name (str, optional): Name of the dataset.
        """
        self.X = X
        self.Y = Y
        if name is not None:
            self.name = name

    def __iter__(self):
        """
        Make the dataset iterable, yielding either (x, y) tuples or just x depending on the presence of Y.
        """
        if self.Y is not None:
            for x, y in zip(self.X, self.Y):
                yield (x, y)
        else:
            for x in self.X:
                yield x

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.X)


# usna streaming: essayer c_nu = d**0.5, d**2/3 et d
