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
        """
        self.k = k                  # number of normal dists
        self.dim = dim              # number of dims in each normal dist
        if a is None:               # prior of each normal dist (sum to 1)
            a = np.random.random(k)
        else:
            a = np.array(a)
        self.a = a / a.sum()
        self._cum = np.cumsum(self.a)

        if mu is None:              # mu for each normal dist, col-wise shape=(dim, k)
            self.mu = np.random.random((dim, k)) * 2 - 1
        else:
            self.mu = np.array(mu)
            assert self.mu.shape == (dim, k)

        if sigma is None:           # sigma for each normal dist, layer-wise shape=(dim, dim, k)
            self.sigma = np.dstack([rand_positive(dim) for _ in range(k)])
        else:
            self.sigma = np.array(sigma)
            assert self.sigma.shape == (dim, dim, k)

        self.index = None           # class of each sample from last observation
        self.data = None            # last observation

    def _choose(self, n):
        """
        Choose n indexes from [0, k-1] according to prior a. (Result also saved in self.index)
        :param n: number of samples
        :return: [int] * n
        """
        rand = np.random.rand(n)
        ret = [0] * n
        for i in range(n):
            ret[i] = np.argwhere(np.array(rand[i] < self._cum, dtype=bool))[0][0]
        self.index = np.array(ret)
        return self.index

    def observe(self, n):
        """
        Get n randomly generated observations from GMM model (row-wise)
        :return: np.array, shape=(n, dim)
        """
        index = self._choose(n)
        data = np.zeros((n, self.dim))
        for i, v in enumerate(index):
            data[i, :] = np.random.multivariate_normal(self.mu[:, v], self.sigma[:, :, v])
        self.data = data
        return data

    def get_hidden_var(self):
        return self.index.copy()

    def plot(self):
        if self.dim == 2:
            return self.plot_scatter()
        elif self.dim == 1:
            return self.plot_hist()

    def plot_hist(self):
        from matplotlib import pyplot as plt, cm
        assert self.dim == 1
        plt.hist(self.data, 30, color='g')
        return plt

    def plot_scatter(self):
        from matplotlib import pyplot as plt, cm
        assert self.dim == 2
        colors = cm.rainbow(np.linspace(0, 0.85, self.k))
        for i in range(self.k):
            xy = self.data[self.index == i, :]
            plt.scatter(xy[:, 0], xy[:, 1], alpha=0.4, color=colors[i], marker='+')
        return plt
