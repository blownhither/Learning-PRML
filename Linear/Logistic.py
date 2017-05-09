import numpy as np
from scipy.linalg.misc import norm


class LogisticRegressionClassifier():
    def __init__(self):
        self.n_features = None
        self.n_samples = None
        self.w = None
        self.y = None
        self.x = None
        self.last_cost = 0
        self.reg_ratio = None

    def cost(self, reg=True):
        wx = np.dot(self.x, self.w)
        wx = np.clip(wx, -500, 500)
        ret1 = -np.sum(wx * self.y)
        ret2 = np.sum(np.log(1 + np.exp(wx)))

        if reg is True:
            # TODO: regularization in this way?
            ret1 += norm(self.w)
        return ret1 + ret2

    def gradient(self):
        wx = np.dot(self.x, self.w)

        wx = np.clip(wx, -500, 500)

        exp = np.exp(wx)
        ret = - np.dot(self.y - exp / (exp + 1.0), self.x)

        ret += self.w / self.reg_ratio

        if np.isnan(ret).any():
            raise Exception('Stop here!')

        return ret

    def _gradient_decrease(self, ratio):
        if self.last_cost is None:
            self.last_cost = self.cost()
        self.w -= ratio * self.gradient()

        if np.isnan(self.w).any():
            raise Exception('Stop here!')

        cost = self.cost()
        ret = cost - self.last_cost
        self.last_cost = cost
        return ret

    def train(self, x, y, reg_ratio=20):
        # use row vector
        x = np.array(x)
        self.n_samples, self.n_features = x.shape
        self.x = np.concatenate((x, np.ones((self.n_samples, 1))), 1)
        self.w = np.random.rand(self.n_features + 1)

        self.reg_ratio = reg_ratio

        self.y = np.array(y)

        assert len(y) == self.n_samples

        for i in range(1000):
            improvement = self._gradient_decrease(0.8)

    def predict(self, x):
        x = np.array(x)
        if x.ndim == 1:
            assert x.shape == (self.n_features,)
            n = 1
        elif x.ndim == 2:
            assert x.shape[1] == self.n_features
            n = x.shape[0]
        else:
            raise Exception()

        x = np.concatenate((x, np.ones((n, 1))), 1)
        wx = np.dot(x, self.w)

        wx = np.clip(wx, -500, 500)

        y = 1 / (1 + np.exp(-wx))
        return y



def test():
    l = LogisticRegressionClassifier()
    l.train([[1, 1], [0, 0]], [1, 0], 9)

    ret = l.predict([[1, 1], [0, 0]])
    print(ret)


if __name__ == '__main__':
    test()