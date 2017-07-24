import numpy as np


class LinearRegressor:
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None

        self.n_features = None
        self.n_samples = None

        self.reg_ratio = None
        self.grad_ratio = None

    def cost(self):
        delta = self.y - np.dot(self.x, self.w)
        c = np.dot(delta, delta)
        c += np.dot(self.w, self.w) / self.reg_ratio / 2.0
        return c

    def gradient(self):
        g = np.dot(self.x.T, np.dot(self.x, self.w) - self.y) * 2
        g += self.w / self.reg_ratio
        return g

    def _gradient_decrease(self):
        self.w -= (self.gradient() * self.grad_ratio)
        print(self.w)

    def train(self, x, y, n_batches=5000, reg_ratio=100, grad_ratio=0.001):
        self.x = np.array(x)
        self.y = np.array(y)

        if not self.x.ndim == 2:
            raise AssertionError("x should be matrix")
        self.n_samples, self.n_features = self.x.shape

        for i in range(self.n_features):
            temp = self.x[:, i]
            temp = (temp - temp.mean() / 2.0) / (temp.max() - temp.min()) * 2
            self.x[:, i] = temp

        self.x = np.concatenate((self.x, np.ones((self.n_samples, 1))), 1)

        self.w = np.zeros(self.n_features + 1)  # TODO: initialization?
        # self.w = np.random.rand(self.n_features + 1)
        self.reg_ratio = reg_ratio
        self.grad_ratio = grad_ratio

        assert self.y.shape[0] == self.n_samples

        for i in range(n_batches):
            self._gradient_decrease()

    def predict(self, x):
        x = np.array(x)
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            if not x.shape == (self.n_features,):
                raise AssertionError("Shape of single-line x should be in shape %d but %s given" % \
                                     (self.n_features, str(x.shape)))
            n = 1
        else:
            if not x.shape[1] == self.n_features:
                raise AssertionError("Shape of matrix x should have %d cols but %d given" % \
                                     (self.n_features, x.shape[1]))
            n = x.shape[0]

        x = np.concatenate((x, np.ones((n, 1))), 1)
        wx = np.dot(x, self.w)
        return wx


def test():
    import pandas as pds
    df = pds.read_csv('Dataset/watermelon-tiny.csv')
    index = np.arange(len(df))
    np.random.shuffle(index)
    df = df.iloc[index]
    x = df[df.columns[1:-1]]
    y = df[df.columns[-1]] - 1

    l = LinearRegressor()
    # a = np.arange(10).reshape((-1, 1))
    # b = np.arange(10)
    # l.train(a, b)
    l.train(x, y)

    pred = l.predict(x)
    pred = (pred > 0.5)

    print(y)
    print(pred)

    accu = np.count_nonzero(pred == np.array(y)) / float(len(pred))
    print(accu)


if __name__ == '__main__':
    test()


