import numpy as np
from math import gamma


class Gibbs:
    def __init__(self, k, dim, data):
        self.k = k
        self.dim = dim
        self.data = np.array(data)
        assert data.shape[1] == dim
        self.n = len(data)

        self.a = 1          # as a_k            # TODO: setting
        self.a0 = self.a * k
        self.k0 = 0.1                           # TODO: ?
        self.kn = self.k0 + self.n
        self.v0 = 1
        self.m0 = np.zeros(dim)                 # TODO: check
        self.mn = (self.k0 * self.m0 + self.n * np.mean(data, 0)) / self.kn
        self.s0 = np.zeros((dim, dim))
        s = np.sum(np.square(data), 1)
        self.sn = self.s0 + s + self.k0 * np.square(self.m0).sum() - self.kn * np.square(self.mn).sum()

        self.z = np.random.randint(0, k, size=self.n)
        self.freq = np.bincount(self.z)
        self.inv_freq = self.n - self.freq

    def _term1(self, omit):
        """
        term1 in Gibbs sampling is $P(z_i=k | z_{\i}, \alpha)$
        """
        self.inv_freq[self.z[omit]] -= 1
        ret = (self.inv_freq + self.a) / (self.n + self.a0 - 1)
        self.inv_freq[self.z[omit]] += 1
        return ret

    def _student(self, mu, sigma, freedom):
        p = sigma.shape[0]
        t1 = gamma((freedom + p) / 2.0)
        t2 = gamma(freedom / 2.0) *

    def _term2(self, omit):
        """
        term2 in Gibbs sampling is $p(x_i | x_{k\i}, \beta)$
        """
