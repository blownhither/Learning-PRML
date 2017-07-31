import numpy as np
from scipy.linalg.misc import norm


class LogisticRegressionClassifier:
    """
    A binary classifier based on Logistic
    """
    def __init__(self):
        self.n_features = None
        self.n_samples = None
        self.w = None
        self.y = None
        self.x = None
        self.last_cost = 0
        self.reg_ratio = None

    def cost(self):
        """
        Cost function with regularization
        :return: float
        """
        wx = np.dot(self.x, self.w)     # row-wise multiplication
        wx = np.clip(wx, -500, 500)     # avoid overflow
        ret1 = -np.sum(wx * self.y)
        ret2 = np.sum(np.log(1 + np.exp(wx)))
        ret1 += norm(self.w)
        return ret1 + ret2

    def gradient(self):
        """
        Gradient on stored x and w
        :return: float
        """
        wx = np.dot(self.x, self.w)
        wx = np.clip(wx, -500, 500)
        exp = np.exp(wx)
        ret = - np.dot(self.y - exp / (exp + 1.0), self.x)
        ret += self.w / self.reg_ratio
        return ret

    def _gradient_decrease(self, ratio):
        """
        Gradient decrease procedure
        :param ratio: float, ratio of gradient applied to this procedure
        :return: float, new cost estimation
        """
        if self.last_cost is None:
            self.last_cost = self.cost()
        self.w -= ratio * self.gradient()
        cost = self.cost()
        ret = cost - self.last_cost
        self.last_cost = cost
        return ret

    def train(self, x, y, n_batches=10000, reg_ratio=100, grad_ratio=0.6):
        """
        Start training process
        :param x: matrix, row-wise samples
        :param y: list, 0 or 1
        :param n_batches: int, number of gradient decrease applied
        :param reg_ratio: int, regularization ratio 1/reg_ratio
        :param grad_ratio: float, ratio of gradient decrease intensity
        :return:
        """
        # use row vector
        x = np.array(x)
        self.n_samples, self.n_features = x.shape
        self.x = np.concatenate((x, np.ones((self.n_samples, 1))), 1)
        self.w = np.zeros(self.n_features + 1)              # TODO: initialization?
        # self.w = np.random.rand(self.n_features + 1)
        self.reg_ratio = reg_ratio
        self.y = np.array(y)
        assert len(y) == self.n_samples
        for _ in range(n_batches):
            self._gradient_decrease(grad_ratio)

    def predict(self, x):
        """
        Start prediction process
        :param x: matrix, row-wise samples
        :return: [float], Possibility P(y=1|x)
        """
        x = np.array(x)
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            assert x.shape == (self.n_features,)
            n = 1
        else:
            assert x.shape[1] == self.n_features
            n = x.shape[0]

        x = np.concatenate((x, np.ones((n, 1))), 1)
        wx = np.dot(x, self.w)
        wx = np.clip(wx, -500, 500)
        y = 1 / (1 + np.exp(-wx))
        return y


def test():
    import pandas as pds
    df = pds.read_csv('Dataset/watermelon-tiny.csv')
    index = np.arange(len(df))
    np.random.shuffle(index)
    df = df.iloc[index]
    x = df[df.columns[1:-1]]
    y = df[df.columns[-1]] - 1

    l = LogisticRegressionClassifier()
    l.train(x, y)
    pred = l.predict(x)
    pred = (pred > 0.5)

    print(y)
    print(pred)

    accu = np.count_nonzero(pred == np.array(y)) / float(len(pred))
    if (pred == 0).all():
        raise Exception('?')
    print(accu)


if __name__ == '__main__':
    for i in range(100):
        test()
