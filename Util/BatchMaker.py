import numpy as np


class BatchMaker:
    def __init__(self, x, y_):
        self._x = np.array(x)
        self._y = np.array(y_)
        # self._used = False

        assert self._x.shape[0] == self._y.shape[0]

        self._p = 0
        self._n = self._x.shape[0]
        self.shuffle_count = 0
        self.shuffle()

    def shuffle(self):
        index = np.arange(self._n)
        np.random.shuffle(index)
        self._x = self._x[index]
        self._y = self._y[index]
        self.shuffle_count += 1

    def next_batch(self, size):
        # self._used = True
        assert 0 < size <= self._n
        if self._p + size > self._n:
            self.shuffle()
            self._p = 0
        ret = (self._x[self._p:self._p + size].copy(), self._y[self._p:self._p + size].copy())
        self._p += size
        return ret

    def all(self):
        self.shuffle()
        self._p = 0
        return self._x.copy(), self._y.copy()

    def split(self, test_ratio=None, n_test=None):
        assert test_ratio is None or n_test is None
        self.shuffle()

        if test_ratio == 0 or n_test == 0:
            return self, None

        end = n_test or int((1 - test_ratio) * self._n)
        assert 0 < end < self._n

        test = BatchMaker(self._x[end:].copy(), self._y[end:].copy())
        self._x = self._x[:end]
        self._y = self._y[:end]
        self._n = end
        return self, test

    def size(self):
        return self._n

    def shape(self):
        return self._x.shape, self._y.shape


