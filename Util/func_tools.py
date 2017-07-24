import numpy as np


def accuracy(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.count_nonzero(x == y) / len(x)


def reindex(x):
    x = np.array(x)
    ret = np.zeros(x.shape, dtype=np.int)
    for idx, value in enumerate(set(x)):
        ret[x == value] = idx
    return ret
