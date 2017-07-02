import numpy as np
from math import gamma
from scipy.special import gammaln
from scipy.stats import multivariate_normal, wishart
import pandas as pds

from Probabilistic.GMM import GMM, rand_positive
from Probabilistic.EM import *


class Gibbs:
    def __init__(self, k, dim, data):
        self.k = k
        self.dim = dim
        self.data = np.array(data)
        assert data.shape[1] == dim
        self.n = len(data)

        self.a = 1.0
        self.a0 = self.a
        self.ak = self.a0 / self.k
        self.k0 = 10.0
        self.kn = self.k0 + self.n
        self.v0 = dim + 5             # (v0 > dim + 1)must holds
        # self.m0 = np.ones(dim)
        # self.s0 = np.ones((dim, dim))
        self.m0 = np.mean(data, 0)
        self.s0 = np.cov(data.T)
        # self.s0 = np.zeros((dim, dim))
        self.z = np.random.randint(0, k, size=self.n)
        self.freq = np.bincount(self.z)

    def _cov(self, data):
        ret = np.zeros((self.dim, self.dim))
        for d in data:
            ret += np.dot(d.reshape((self.dim, 1)), d.reshape((1, self.dim)))
        return ret

    def _cov_single(self, data):
        return np.dot(data.reshape((self.dim, 1)), data.reshape((1, self.dim)))

    def _term1(self, omit):
        """
        term1 in Gibbs sampling is $P(z_i=k | z_{\i}, \alpha)$
        """
        self.freq[self.z[omit]] -= 1
        ret = (self.freq + self.ak) / (self.n + self.a0 - 1)
        self.freq[self.z[omit]] += 1
        return ret

    def _term2(self, k, x_star):
        """
        term2 in Gibbs sampling is $p(x_i | x_{k\i}, \beta)$
        """
        index = np.where(self.z == k)[0]
        if len(index) == 0:
            return 0
        data = self.data[index]
        n = data.shape[0]
        kn = self.k0 + n
        vn = self.v0 + n

        # data_cov = self._cov(data)
        data_cov = np.cov(data.T)
        data_mean = np.mean(data, 0)

        # TODO: this works well but it is not the original scheme in the paper
        return multivariate_normal(data_mean, data_cov).pdf(x_star)

        # TODO: the original scheme in the paper
        # data_mean_star = (np.mean(data, 0) * n + x_star) / (n + 1)
        # mn = (self.k0 * self.m0 + n * data_mean) / kn
        # mn_star = (self.k0 * self.m0 + (n + 1) * data_mean_star) / kn
        # sn = self.s0 + data_cov + self.k0 * self._cov_single(self.m0) + kn * self._cov_single(mn)
        # sn_star = self.s0 + data_cov + self._cov_single(x_star) + self.k0 * self._cov_single(self.m0) + (kn + 1) * self._cov_single(mn_star)
        # numerator1 = np.power(np.pi * (kn + 1) / kn, -self.dim / 2.0)
        # temp = np.linalg.det(sn_star)
        # numerator2 = np.power(temp / np.linalg.det(sn), -vn / 2.0) * np.power(temp, -0.5)
        # numerator3 = np.exp(gammaln((vn + 1) / 2.0) - gammaln((vn + 1 - self.dim) / 2.0))
        # ret = numerator1 * numerator2 * numerator3
        # return ret

    def _form_dist(self):
        """ From status sequence, rebuild k Gaussian distributions """
        mu = np.zeros((self.dim, self.k))
        sigma = np.zeros((self.dim, self.dim, self.k))
        pi = np.zeros(self.k)
        for i in range(self.k):
            c = self.data[self.z == i]
            pi[i] = (len(c) / float(self.n))
            mu[:, i] = c.mean(0)
            sigma[:, :, i] = np.cov(c, rowvar=False)
        return mu, sigma, pi

    # def _form_dist(self):
    #    """ Form k Gaussian distributions with sampling method """
    #     mu = np.zeros((self.dim, self.k))
    #     sigma = np.zeros((self.dim, self.dim, self.k))
    #     _pi = np.bincount(self.z) / float(self.n)
    #     pi = np.zeros(self.k)
    #     pi[:len(_pi)] = _pi
    #     for i in range(self.k):
    #         mu[:, i], sigma[:, :, i] = self._update_component(i)
    #     return mu, sigma, pi

    def test(self):
        """ Routine API: print likelihood """
        mu, sigma, pi = self._form_dist()
        self.log_likelihood(mu, sigma, pi)

    def fit_and_test(self, max_iter):
        """ Combine fit and test routine """
        for _iter in range(max_iter):
            if _iter % 1 == 0:
                self.test()
                mu, sigma, pi = self._form_dist()
                print(self.log_likelihood(mu, sigma, pi))
            for i in range(self.n):
                t1 = self._term1(i)

                t2 = np.zeros(self.k)
                last = self.z[i]
                self.freq[last] -= 1
                self.z[i] = -1
                for k in range(self.k):
                    t2[k] = self._term2(k, self.data[i])
                cum = np.cumsum(t1 * t2)
                cum /= cum[-1]

                k_new = np.where(np.random.random() < cum)[0][0]
                self.z[i] = k_new
                self.freq[k_new] += 1

    def predict(self):
        return self._form_dist()

    def log_likelihood(self, mu, sigma, pi):
        mat = np.zeros((self.n, self.k))
        for i in range(self.k):
            dist = multivariate_normal(mu[:, i], sigma[:, :, i])
            for j in range(self.n):
                mat[j, i] = dist.pdf(self.data[j])
        evaluation = np.sum(np.log(np.dot(mat, pi)))
        print(evaluation)
        return evaluation


def test():
    g = GMM(5, 2)
    data = g.observe(1000)
    gb = Gibbs(5, 2, data)
    gb.fit_and_test(100)


if __name__ == '__main__':
    test()


