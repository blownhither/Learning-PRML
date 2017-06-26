import numpy as np
# from matplotlib import pyplot as plt


def rand_positive(n):
    """
    Generate positive semi-definite matrix with size (n, n)
    :param n: mat size
    :return: np.array
    """
    ret = np.random.random((n, n))
    return np.dot(ret.T, ret)


class GMM:
    def __init__(self, k=2, dim=2, a=None, mu=None, sigma=None):
        """
        A GMM model generating data with distribution of $\sum{a_k \phi(y|\theta_k)}$
        :param k:
        """
        self.k = k
        self.dim = dim
        if a is None:
            a = np.random.random(k)
        else:
            a = np.array(a)
        self.a = a / a.sum()
        self._cum = np.cumsum(self.a)

        if mu is None:
            self.mu = np.random.random((dim, k))
        else:
            self.mu = np.array(mu)
            assert self.mu.shape == (dim, k)

        if sigma is None:
            self.sigma = np.dstack([rand_positive(dim) for _ in range(k)])
        else:
            self.sigma = np.array(sigma)
            assert self.sigma.shape == (dim, dim, k)
        self.sigma2 = np.square(self.sigma)

    def _choose(self, n):
        rand = np.random.rand(n)
        ret = [0] * n
        for i in range(n):
            ret[i] = np.argwhere(rand[i] < self._cum)[0][0]
        return np.array(ret)

    def observe(self, n):
        """
        Get n randomly generated observations from GMM model
        :return: np.array, shape=(n, dim)
        """
        index = self._choose(n)
        ret = np.zeros((n, self.dim))
        for i in index:
            ret[i] = np.random.multivariate_normal(self.mu[:, i], self.sigma[:, :, i])
        return ret

    # def observe_and_plot(self, n):
    #     index = self._choose(n)
    #     data = np.random.normal(self.mu[index], self.sigma[index])
    #     plt.hist(data[index == 0], alpha=0.6)
    #     plt.hist(data[index == 1], alpha=0.6)
    #     plt.show()
    #     return data


class EM(GMM):
    def __init__(self, k, dim, data):
        """
        Here EM is a GMM model undergoing iterative updating
        :param k:
        :param data:
        """
        super().__init__(k, dim)
        self.data = data
        assert data.shape[1] == dim

        self.n = len(data)

    def _gaussian_density(self, single_data):
        """
        :param single_data: should be ONE single data
        :return:
        """
        # return np.exp(- np.square(single_data - self.mu) / 2.0 / self.sigma2) / np.sqrt(np.pi * 2) / self.sigma
        ret = np.zeros(self.k)
        for i in range(self.k):
            temp1 = np.power(np.linalg.det(2 * np.pi * self.sigma[:, :, i]), -0.5)
            temp2 = np.mat(single_data - self.mu[:, i])         # as row
            temp3 = np.exp(-0.5 * temp2 * np.linalg.pinv(self.sigma[:, :, i]) * temp2.T)
            ret[i] = temp1 * temp3
        return ret


    def expectation(self):
        """
        :return: response[data_j, model_k]
        """
        response = np.zeros((self.n, self.k))
        for j in range(self.n):
            res = self.a * self._gaussian_density(self.data[j])
            response[j] = res / res.sum()
        return response

    def maximization(self, response):
        r_sum = np.sum(response, 0)
        mu = np.dot(response.T, self.data) / r_sum
        sigma2 = np.zeros((self.dim, self.dim, self.k))
        for i in range(self.k):
            sigma2[i] = np.dot(response[:, i], np.square(self.data - mu[:, i]))
            # sigma2[i] = np.dot(response[:, i], np.square(self.data - self.mu[:, i]))
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
            if i % 10 == 0:
                err_mu = np.linalg.norm(self.mu - mu)
                err_sigma = np.linalg.norm(self.sigma - sigma)
                err_a = np.linalg.norm(self.a - a)
                print(i, ":", err_mu, err_sigma, err_a)


def test_gmm():
    g = GMM(5)
    o = g.observe(1000000)
    prec = np.mean(o) - np.dot(g.a, g.mu)
    print(prec)


def test_em():
    e = np.eye(2, 2)
    g = GMM(2, 2)
    # g.observe_and_plot(1000)
    data = g.observe(20000)
    e = EM(2, 2, data)
    e.fit_and_test(10000, g.mu, g.sigma, g.a)

if __name__ == '__main__':
    test_em()
