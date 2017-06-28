import numpy as np

from Probabilistic.GMM import GMM
"""
This file runs a EM algorithm on GMM model and provides several visualization functions.
"""


class EM(GMM):
    def __init__(self, k, dim, data):
        """
        Here EM is a GMM model undergoing iterative updating
        :param k:   should be same with GMM observed
        :param dim: should be same with GMM observed
        :param data:observations, [float] * (n, dim)
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
        ret = np.zeros(self.k)
        for i in range(self.k):
            det = np.linalg.det(np.array(2 * np.pi * self.sigma[:, :, i]))
            temp1 = np.power(np.clip(det, 1e-100, 1e100), -0.5)
            temp2 = np.mat(single_data - self.mu[:, i])
            temp3 = np.exp(-0.5 * temp2 * np.linalg.pinv(self.sigma[:, :, i]) * temp2.T)
            ret[i] = temp1 * temp3
            # assert not np.any(np.isnan(ret[i])), (det, temp1, temp2, temp3, ret[i])
        return ret

    def expectation(self):
        """
        Expectation step of EM algorithm, return response
        :return: response[data_j, model_k]
        """
        response = np.zeros((self.n, self.k))
        for j in range(self.n):
            res = self.a * self._gaussian_density(self.data[j])
            response[j, :] = res / np.clip(res.sum(), 1e-100, 1e100)
            # assert not np.any(np.isnan(response[j])), (res, response[j])
        return response

    def maximization(self, response):
        """
        Maximization step of EM algorithm, update param
        :param response: given by expectation()
        :return: no return
        """
        r_sum = np.sum(response, 0)
        # assert np.all(r_sum > 0), (r_sum, response)
        r_sum = np.clip(r_sum, 1e-100, 1e100)

        # calculate new value of normal dist params
        mu = np.dot(self.data.T, response) / r_sum
        sigma = np.zeros((self.dim, self.dim, self.k))
        for i in range(self.k):
            temp = np.zeros((self.dim, self.dim))
            for j in range(self.n):
                centered = (self.data[j] - mu[:, i]).reshape((self.dim, 1))     # based on new mu, not old mu
                temp += response[j, i] * np.outer(centered, centered)
            sigma[:, :, i] = temp / r_sum[i]
        a = r_sum / self.n

        # update stored dist param
        self.mu = mu
        self.sigma = sigma
        self.a = a

    def fit(self, n_iter=100):
        """ Dummy fit """
        for i in range(n_iter):
            self.maximization(self.expectation())

    def test(self, mu, sigma, a):
        """ Calculate norm diff between stored dist param and given truth """
        err_mu = np.linalg.norm(self.mu - mu)
        err_sigma = np.linalg.norm(self.sigma - sigma)
        err_a = np.linalg.norm(self.a - a)
        print(err_mu, err_sigma, err_a)

    def fit_and_test(self, n_iter, mu, sigma, a):
        """ Fit and print accuracy a few rounds """
        for i in range(n_iter):
            if i % 5 == 0:
                print("iter ", i)
                self.test(mu, sigma, a)
                print(self.log_likelihood())
            self.maximization(self.expectation())

    def log_likelihood(self):
        """ Log likelihood for evaluation """
        ret = 0
        for i in range(self.n):
            temp = np.dot(self.a, self._gaussian_density(self.data[i]))
            ret += np.log(np.clip(temp, 1e-100, 1e100))
        return ret


def test_gmm():
    """ Unittest """
    g = GMM(5, 1)
    o = g.observe(10000)
    precision = np.mean(o, 0) - np.dot(g.a, g.mu.T)
    print(precision)


def plot_hidden(g, data):
    """
    scatter plot 2-d samples, colored by classes
    :return plt module reference
    """
    from matplotlib import pyplot as plt, cm
    assert g.dim == 2
    colors = cm.rainbow(np.linspace(0, 0.85, g.k))
    index = g.get_hidden_var()
    for i in range(g.k):
        xy = data[index == i, :]
        plt.scatter(xy[:, 0], xy[:, 1], alpha=0.4, color=colors[i], marker='+')
    return plt


def plot_dist(mu, sigma):
    """ Contour plot of one normal distribution """
    sigma = np.array(sigma)
    mu = np.array(mu)

    from matplotlib import pyplot as plt, mlab
    delta = 0.025
    x = np.arange(-1, 1, delta)
    y = np.arange(-1, 1, delta)
    axis_x, axis_y = np.meshgrid(x, y)
    dist = mlab.bivariate_normal(axis_x, axis_y, mux=mu[0], muy=mu[1],
                                 sigmax=sigma[0, 0], sigmay=sigma[1, 1], sigmaxy=sigma[0, 1] ** 2)
    contour = plt.contour(axis_x, axis_y, dist, 5)
    plt.clabel(contour, inline=1, fontsize=10)
    return plt


def test_em():
    """ Unittest """
    g = GMM(5, 2)
    data = g.observe(1000)
    # draw_hidden(g, data)
    # plt = draw_dist(g.mu[:, 0], g.sigma[:, :, 0])
    # plt.show()
    # g.observe_and_plot(1000)
    e = EM(5, 2, data)
    e.fit_and_test(1000, g.mu, g.sigma, g.a)


if __name__ == '__main__':
    test_em()
