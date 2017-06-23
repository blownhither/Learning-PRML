import numpy as np


class GMM:
    def __init__(self, k):
        """
        A GMM model generating data with distribution of $\sum{a_k \phi(y|\theta_k)}$
        :param k:
        """
        self.k = k
        a = np.random.rand(k)
        self.a = a / a.sum()
        self.mu = np.random.rand(k)
        self.sigma = np.random.rand(k)
        self._cum = np.cumsum(self.a)

    def _choose(self, n):
        rand = np.random.rand(n)
        ret = [0] * n
        for i in range(n):
            ret[i] = np.argwhere(rand[i] < self._cum)[-1][0]
        return ret

    def observe(self, n):
        """
        Get n randomly generated observations from GMM model
        :return: np.array, shape=(n,)
        """
        index = self._choose(n)
        return np.random.normal(self.mu[index], self.sigma[index])


class EM(GMM):
    def __init__(self, k, data):
        """
        EM is a GMM model undergoing iterative updating
        :param k:
        :param data:
        """
        super().__init__(k)



def test_GMM():
    g = GMM(5)
    o = g.observe(1000000)
    prec = np.mean(o) - np.dot(g.a, g.mu)
    print(prec)

if __name__ == '__main__':
    test_GMM()
