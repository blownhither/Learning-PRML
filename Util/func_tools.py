import numpy as np


def accuracy(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.count_nonzero(x == y) / len(x)