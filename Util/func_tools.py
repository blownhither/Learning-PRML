import numpy as np
from collections import Counter


def accuracy(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.count_nonzero(x == y) / len(x)


def reindex(x):
    x = np.array(x)
    ret = np.zeros(x.shape)
    for idx, key in enumerate(set(x)):
        ret[x == key] = idx
    return ret
