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
        self.k0 = 1.0               # TODO: ?
        self.kn = self.k0 + self.n
        self.v0 = k + 2             # (v0 > dim + 1)must holds
        # self.vn = self.v0 + self.n
        self.m0 = np.ones(dim)     # TODO: check
        # self.mn = (self.k0 * self.m0 + self.n * np.mean(data, 0)) / self.kn
        self.s0 = np.array([[1, 0], [0, 1]])
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
        self.freq[self.z[omit]] -= 1
        ret = (self.freq + self.ak) / (self.n + self.a - 1)
        self.freq[self.z[omit]] += 1
        return ret

        # def _student(self, mu, sigma, freedom):
        # p = sigma.shape[0]
        # t1 = gamma((freedom + p) / 2.0)
        # t2 = gamma(freedom / 2.0) * np.power(freedom * np.pi, p / 2.0) * np.sqrt(np.linalg.det(sigma))
        # center =
        # t3 = np.power(1 + )

    def _term2(self, k, x_star):
        """
        term2 in Gibbs sampling is $p(x_i | x_{k\i}, \beta)$
        """
        index = np.where(self.z == k)
        data = self.data[index]
        n = data.shape[0]
        kn = self.k0 + n
        vn = self.v0 + n
        sn = self.s0 + self._cov(data) + self._cov_single(data.mean(0) - self.m0) * self.k0 * n / (self.k0 + n)
        sn_star = sn + self._cov_single(x_star)

        # serial = list(map(gamma, np.arange((vn + 1 - self.dim) / 2.0, (vn + 1.1) / 2.0, 0.5)))
        # assert len(serial) == self.dim + 1
        # prod = np.prod(serial)

        numerator1 = np.power(np.pi, -self.dim / 2.0)
        numerator3 = np.prod(np.arange((vn + 1 - self.dim) / 2.0, vn / 2.0 - 1, 1))
        # numerator2 = np.power(np.linalg.det(sn_star), -(vn + 1) / 2.0)
        # denominator2 = np.power(np.linalg.det(sn), -vn / 2.0)
        temp = np.linalg.det(sn_star)
        numerator2 = np.power(temp / np.linalg.det(sn), -vn / 2.0) * np.power(temp, -0.5)
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
        err_mu = np.linalg.norm(mu_t - mu)
        err_sigma = np.linalg.norm(sigma_t - sigma)
        err_pi = np.linalg.norm(pi_t - pi)
        print(err_mu, err_sigma, err_pi)
        return err_mu, err_sigma, err_pi

    def fit_and_test(self, max_iter, mu_t, sigma_t, pi_t):
        for _iter in range(max_iter):
            if _iter % 5 == 0:
                print(_iter)
                self.test(mu_t, sigma_t, pi_t)
                self.log_likelihood()

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

                k_new = np.where(np.random.random() < cum)[0][0]
                if k_new != last:
                    count += 1
                self.z[i] = k_new
                self.freq[k_new] += 1

                assert self.n == self.freq.sum()
            print(count)

    def log_likelihood(self):
        mu, sigma, pi = self._form_dist()
        mat = np.zeros((self.n, self.k))
        for i in range(self.k):
            dist = multivariate_normal(mu[:, i], sigma[:, :, i])
            for j in range(self.n):
                mat[j, i] = dist.pdf(self.data[j])
        evaluation = np.sum(np.log(np.dot(mat, pi)))
        print(evaluation)




def test():
    g = GMM(3, 2)
    data = g.observe(500)
    gb = Gibbs(3, 2, data)
    gb.fit_and_test(1000, g.mu, g.sigma, g.a)

if __name__ == '__main__':
    test()


