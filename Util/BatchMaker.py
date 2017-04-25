import numpy as np


class BatchMaker:
    def __init__(self, x, y_):
        self._x = np.array(x)
        self._y = np.array(y_)

        assert self._x.shape[0] == self._y.shape[0]

        self.p = 0
        self.n = self._x.shape[0]
        self.shuffle()

    def shuffle(self):
        index = np.arange(self.n)
        np.random.shuffle(index)
        self._x = self._x[index]
        self._y = self._y[index]

    def next_batch(self, size):
        assert size < self.n
        if self.p + size > self.n:
            self.shuffle()
            self.p = 0
        ret = (self._x[self.p:self.p+size].copy(), self._y[self.p:self.p+size].copy())
        self.p += size
        return ret



