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
        self.sigma2 = np.square(self.sigma)
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
        Here EM is a GMM model undergoing iterative updating
        :param k:
        :param data:
        """
        super().__init__(k)
        self.data = data
        self.n = len(data)

    def _gaussian_density(self, single_data):
        """
        :param single_data: should be ONE single data
        :return:
        """
        return self.sigma * np.exp(- np.square(single_data - self.mu) / 2.0 / self.sigma2) / np.sqrt(np.pi * 2) / self.sigma

    def expectation(self):
        """
        :return: response[data_j, model_k]
        """
        response = np.zeros((self.n, self.k))
        for i in range(self.n):
            res = self.a * self._gaussian_density(self.data[i])
            response[i] = res / res.sum()
        return response

    def maximization(self, response):
        r_sum = np.sum(response, 0)
        mu = np.dot(response.T, self.data) / r_sum
        sigma2 = np.zeros(self.k)
        for i in range(self.k):
            sigma2[i] = np.dot(response[:, i], np.square(self.data - self.mu[i]))
        sigma2 /= r_sum
        a = r_sum / self.n

        self.mu = mu
        self.sigma2 = sigma2
        self.sigma = np.sqrt(sigma2)
        self.a = a

    def fit(self, n_iter=100):
        for i in range(n_iter):
            self.maximization(self.expectation())

    def test(self, mu, sigma, a):
        err_mu = np.linalg.norm(self.mu - mu)
        err_sigma = np.linalg.norm(self.sigma - sigma)
        err_a = np.linalg.norm(self.a - a)
        print(err_mu, err_sigma, err_a)

    def fit_and_test(self, n_iter, mu, sigma, a):
        for i in range(n_iter):
            self.maximization(self.expectation())
            err_mu = np.linalg.norm(self.mu - mu)
            err_sigma = np.linalg.norm(self.sigma - sigma)
            err_a = np.linalg.norm(self.a - a)
            print(err_mu, err_sigma, err_a)


def test_GMM():
    g = GMM(5)
    o = g.observe(1000000)
    prec = np.mean(o) - np.dot(g.a, g.mu)
    print(prec)


def test_EM():
    g = GMM(2)
    data = g.observe(20000)
    e = EM(2, data)
    e.fit_and_test(10000, g.mu, g.sigma, g.a)

if __name__ == '__main__':
    test_EM()
