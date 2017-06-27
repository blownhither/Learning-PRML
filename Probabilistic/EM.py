import numpy as np


def rand_positive(n):
    """
    Generate positive semi-definite matrix with size (n, n)
    :param n: mat size
    :return: np.array
    """
    ret = (np.random.random((n, n)) - 0.5) * 2
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

        self.index = None

    def _choose(self, n):
        rand = np.random.rand(n)
        ret = [0] * n
        for i in range(n):
            ret[i] = np.argwhere(rand[i] < self._cum)[0][0]
        self.index = np.array(ret)
        return self.index

    def observe(self, n):
        """
        Get n randomly generated observations from GMM model (row-wise)
        :return: np.array, shape=(n, dim)
        """
        index = self._choose(n)
        ret = np.zeros((n, self.dim))
        for i, v in enumerate(index):
            ret[i, :] = np.random.multivariate_normal(self.mu[:, v], self.sigma[:, :, v])
        return ret

    def get_hidden_var(self):
        return self.index.copy()

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
        Note: could possibly return 0
        :param single_data: should be ONE single data (dim,)
        :return:
        """
        # return np.exp(- np.square(single_data - self.mu) / 2.0 / self.sigma2) / np.sqrt(np.pi * 2) / self.sigma
        ret = np.zeros(self.k)
        for i in range(self.k):
            det = np.linalg.det(2 * np.pi * self.sigma[:, :, i])

            temp1 = np.power(np.clip(det, 1e-100, 1e100), -0.5)         # TODO: use clip?
            temp2 = np.mat(single_data - self.mu[:, i])                 # as row
            temp3 = np.exp(-0.5 * temp2 * np.linalg.pinv(self.sigma[:, :, i]) * temp2.T)
            ret[i] = temp1 * temp3

            assert not np.any(np.isnan(ret[i])), (det, temp1, temp2, temp3, ret[i])

        return ret

    def expectation(self):
        """
        :return: response[data_j, model_k]
        """
        response = np.zeros((self.n, self.k))
        for j in range(self.n):
            res = self.a * self._gaussian_density(self.data[j])
            response[j, :] = res / np.clip(res.sum(), 1e-100, 1e100)    # TODO: use clip?

            assert not np.any(np.isnan(response[j])), (res, response[j])

        return response

    def maximization(self, response):
        r_sum = np.sum(response, 0)
        # assert np.all(r_sum > 0), (r_sum, response)
        r_sum = np.clip(r_sum, 1e-100, 1e100)

        mu = np.zeros((self.dim, self.k))
        for i in range(self.k):
            mu[:, i] = np.dot(self.data.T, response[:, i]) / r_sum[i]

        sigma2 = np.zeros((self.dim, self.dim, self.k))
        for i in range(self.k):
            temp = np.zeros((self.dim, self.dim))
            for j in range(self.n):
                centered = (self.data[j] - mu[:, i]).reshape((self.dim, 1))
                temp += response[j, i] * np.dot(centered, centered.T)
            sigma2[:, :, i] = temp / r_sum[i]

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
        # self.test(mu, sigma, a)
        for i in range(n_iter):
            self.maximization(self.expectation())
            if i % 1 == 0:
                # self.test(mu, sigma, a)
                print(self.log_likelihood())

    def log_likelihood(self):
        """
        Log likelihood for evaluation
        :return: double
        """
        ret = 0
        for i in range(self.n):
            temp = np.dot(self.a, self._gaussian_density(self.data[i]))
            ret += np.log(np.clip(temp, 1e-100, 1e100))
        return ret


def test_gmm():
    g = GMM(5, 1)
    o = g.observe(10000)
    prec = np.mean(o, 0) - np.dot(g.a, g.mu.T)
    print(prec)


def draw_hidden(g, data):
    from matplotlib import pyplot as plt, cm
    color = cm.rainbow(np.linspace(0, 0.85, g.k))
    index = g.get_hidden_var()
    for i in range(g.k):
        xy = data[index == i, :]
        plt.scatter(xy[:, 0], xy[:, 1], alpha=0.4, color=color[i], marker='+')
    plt.show()


def test_em():
    # e = np.eye(2, 2)
    g = GMM(3, 2)
    # g.observe_and_plot(1000)
    data = g.observe(10000)
    draw_hidden(g, data)
    # e = EM(3, 2, data)
    # e.fit_and_test(1000, g.mu, g.sigma, g.a)

if __name__ == '__main__':
    test_em()
