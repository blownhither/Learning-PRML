import numpy as np
from math import gamma
from scipy.stats import multivariate_normal

from Probabilistic.GMM import GMM, rand_positive


class Gibbs:
    def __init__(self, k, dim, data):
        self.k = k
        self.dim = dim
        self.data = np.array(data)
        assert data.shape[1] == dim
        self.n = len(data)

        self.a = 1.0                # TODO: setting
        self.a0 = self.a
        self.ak = self.a0 / self.k
        self.k0 = 5.0               # TODO: ?
        self.kn = self.k0 + self.n
        self.v0 = k + 3             # (v0 > dim + 1)must holds
        # self.vn = self.v0 + self.n
        # self.m0 = np.ones(dim)     # TODO: check
        self.m0 = np.mean(data, 0)
        # self.mn = (self.k0 * self.m0 + self.n * np.mean(data, 0)) / self.kn
        self.s0 = np.eye(dim)
        # s = self._cov(data)
        # self.sn = self.s0 + s + self.k0 * self.n / (self.k0 + self.n) * self._cov(
        #     (np.mean(data, 0) - self.m0).reshape((1, -1)))
        # self.sn_det = np.power(np.linalg.det(self.sn), -self.vn / 2.0)

        self.z = np.random.randint(0, k, size=self.n)
        self.freq = np.bincount(self.z)
        # self.inv_freq = self.n - self.freq

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
        # ret = np.ones(self.k, dtype=np.float) / (self.n - 1 + self.a0)
        # component = self.z[omit]
        # ret[component] *= self.freq[component] - 1 + self.ak
        index = self.z[omit]
        self.freq[index] -= 1
        ret = (self.freq + self.ak) / (self.n + self.a - 1)
        self.freq[index] += 1
        return ret

    def _term2(self, k, x_star):
        """
        term2 in Gibbs sampling is $p(x_i | x_{k\i}, \beta)$
        """
        index = (self.z == k)
        if np.count_nonzero(index) == 0:
            return 0
        data = self.data[index]
        n = data.shape[0]
        kn = self.k0 + n
        vn = self.v0 + n
        data_cov = self._cov(data)
        data_mean = np.mean(data, 0)
        sn = self.s0 + data_cov + self._cov_single(data_mean - self.m0) * self.k0 * n / (self.k0 + n)

        assert not np.isnan(sn).any(), (self.s0, data_cov, n, data, index)

        data_mean_star = (data_mean * n + x_star) / (n + 1)
        sn_star = self.s0 + data_cov + self._cov_single(x_star) + self._cov_single(data_mean_star - self.m0) * self.k0 * n / (self.k0 + n)

        numerator1 = np.power(np.pi * (kn + 1) / kn, -self.dim / 2.0)
        temp = np.linalg.det(sn_star)
        numerator2 = np.power(temp / np.linalg.det(sn), -vn / 2.0) * np.power(temp, -0.5)

        assert not np.isnan(numerator2), (sn, sn_star, data_cov, self.z, k)
        assert self.dim % 2 == 0
        # numerator3 = gamma((vn + 1) / 2.0) / gamma((vn + 1 - self.dim) / 2.0)
        numerator3 = np.prod((vn - np.arange(1, 1.1 + self.dim, 2)) / 2)

        assert not np.isnan(numerator1 * numerator2 * numerator3), (numerator1, numerator2, numerator3)

        return numerator1 * numerator2 * numerator3

    # def fit(self, max_iter):
    #     for _iter in range(max_iter):
    #         for i in range(self.n):
    #             t1 = self._term1(i)
    #
    #             t2 = np.zeros(self.k)
    #             self.freq[self.z[i]] -= 1
    #             last = self.z[i]
    #             self.z[i] = -1                                  # take it out
    #             for k in range(self.k):
    #                 t2[k] = self._term2(k, self.data[i])        # TODO: not right
    #
    #             cum = np.cumsum(t1 * t2)
    #             cum /= cum[-1]
    #             k_new = np.where(np.random.random() < cum)[0][0]
    #             self.z[i] = k_new
    #             self.freq[k_new] += 1
    #
    #             assert self.n == self.freq.sum()

    def _form_dist(self):
        mu = np.zeros((self.dim, self.k))
        sigma = np.zeros((self.dim, self.dim, self.k))
        pi = np.zeros(self.k)
        for i in range(self.k):
            c = self.data[self.z == i]
            pi[i] = (len(c) / float(self.n))
            mu[:, i] = c.mean(0)
            sigma[:, :, i] = np.cov(c, rowvar=False)
        return np.array(mu), np.array(sigma), np.array(pi)

    def test(self, mu_t, sigma_t, pi_t):
        mu, sigma, pi = self._form_dist()
        print(mu)
        err_mu = np.linalg.norm(mu_t - mu)
        err_sigma = np.linalg.norm(sigma_t - sigma)
        err_pi = np.linalg.norm(pi_t - pi)
        print(err_mu, err_sigma, err_pi)
        self.log_likelihood(mu, sigma, pi)
        return err_mu, err_sigma, err_pi

    def fit_and_test(self, max_iter, mu_t, sigma_t, pi_t):
        for _iter in range(max_iter):
            if _iter % 5 == 0:
                print('\n%d' % _iter)
                self.test(mu_t, sigma_t, pi_t)

            count = 0
            for i in range(self.n):
                t1 = self._term1(i)

                t2 = np.zeros(self.k)
                self.freq[self.z[i]] -= 1
                last = self.z[i]
                self.z[i] = -1                                  # take it out
                for k in range(self.k):
                    t2[k] = self._term2(k, self.data[i])        # TODO: not right

                cum = np.cumsum(t1 * t2)
                cum /= cum[-1]

                assert cum[-1] == 1.0, (t1, t2)

                k_new = np.where(np.random.random() < cum)[0][0]
                if k_new != last:
                    count += 1
                self.z[i] = k_new
                self.freq[k_new] += 1

                assert self.n == self.freq.sum()
            print(count, end=',')

    def log_likelihood(self, mu, sigma, pi):
        mat = np.zeros((self.n, self.k))
        for i in range(self.k):
            dist = multivariate_normal(mu[:, i], sigma[:, :, i])
            for j in range(self.n):
                mat[j, i] = dist.pdf(self.data[j])
        evaluation = np.sum(np.log(np.dot(mat, pi)))
        print(evaluation)




def test():
    g = GMM(2, 2)
    # g = GMM(2, 2, mu=np.array([[1, 1], [-1, -1]]), sigma=np.dstack([(np.eye(2)), np.eye(2)]))
    data = g.observe(500)
    # data = np.array([[10, 10], [-10, -10]]).repeat(20, 0)
    gb = Gibbs(2, 2, data)
    print('Target log_likelihood: ')
    gb.log_likelihood(g.mu, g.sigma, g.a)
    gb.fit_and_test(1000, g.mu, g.sigma, g.a)

if __name__ == '__main__':
    test()


