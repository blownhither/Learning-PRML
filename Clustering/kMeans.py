#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/27/17 8:45 PM
"""
import numpy as np


class KMeans:
    """
    Simple k-means, using square distance
    """
    def __init__(self, data):
        """
        Feed data matrix
        :param data:
        :param k: number of cluster expected
        """
        data = np.array(data, dtype=np.float)
        self.n_data = data.shape[0]
        self.dim = data.shape[1]
        self._scale = np.mean(data, 0), np.std(data, 0)
        self.data = (data - self._scale[0]) / self._scale[1]    # data normalized
        self.centers = None
        self.labels = None
        self.k = None
        self.colors = None

    def fit(self, k, max_iter=1000, plot=False):
        """
        Fitting k-means on given data
        :param k: number of clusters
        :param max_iter: Max loops
        :param plot: plot data point (centered to [-1, 1])
        """
        self.k = k
        self.centers = np.random.random((k, self.dim))
        last_dbi = -1
        if plot:
            from matplotlib import pyplot as plt, cm
            plt.ion()
            self.colors = cm.rainbow(np.linspace(0, 0.70, self.k))
        for iter_ in range(max_iter):
            self.labels = self._iter()
            for i in range(k):
                self.centers[i] = np.mean(self.data[self.labels == i], 0)

            if iter_ % 5 == 0:
                dbi = self._db_index()
                print("Iter %d, Davies-Bouldin Index: %g" % (iter_, dbi))
                if last_dbi == dbi:
                    break
                last_dbi = dbi
                if plot:
                    self._plot()

    def _iter(self):
        """
        Perform one iteration
        :return: new labels assigned to each data
        """
        dist = np.zeros((self.n_data, self.k))
        for i in range(self.k):
            dist[:, i] = np.linalg.norm(self.data - self.centers[i], 2, 1)
        return np.argmin(dist, 1)

    @staticmethod
    def _cluster_avg(cluster):
        """
        Average distance inside clusters
        :param cluster: subset of data
        :return: average distance
        """
        sum_ = 0.0
        n_ = len(cluster)
        if n_ == 0:
            return
        for i in range(n_):
            for j in range(i + 1, n_):
                sum_ += np.linalg.norm(cluster[i] - cluster[j])
        return 2 * sum_ / n_ / (n_ + 1)

    def _db_index(self):
        """
        Davies-Bouldin Index of a clustering scheme
        :return: db_index
        """
        avg = np.array([self._cluster_avg(self.data[self.labels == i]) for i in range(self.k)])
        dbi = -np.ones((self.k, self.k))
        for i in range(self.k):
            for j in range(i + 1, self.k):
                temp = (avg[i] + avg[j]) / np.linalg.norm(self.centers[i] - self.centers[j])
                dbi[i, j] = temp
                dbi[j, i] = temp
        dbi = np.max(dbi, 1)
        return float(np.sum(dbi) / self.k)

    def _plot(self):
        """
        Plot scatter points with different colors
        """

        # TODO: support PCA
        from matplotlib import pyplot as plt
        for i in range(self.k):
            points = self.data[self.labels == i]
            plt.scatter(points[:, 0], points[:, 1], color=self.colors[i])
            plt.scatter(self.centers[i, 0], self.centers[i, 1], 100, marker='+', color='red')
        plt.show()
        plt.pause(0.1)
        plt.clf()

    def get_centers(self):
        return (self.centers * self._scale[1]) + self._scale[0]

    def get_labels(self):
        return self.labels.copy()

    def predict(self, data):
        """
        Predict labels assigned to data with fitted model
        :param data: same shape as fitting data
        :return:
        """
        if data.ndim == 1:
            return np.argmin(np.linalg.norm(self.centers - data, 2, 1))
        labels_ = np.zeros(len(data))
        for i, v in enumerate(data):
            labels_[i] = np.argmin(np.linalg.norm(self.centers - v, 2, 1))
        return labels_


def _test_k_means():
    k = KMeans(np.random.random((1000, 2)))
    k.fit(4, plot=True)


if __name__ == '__main__':
    _test_k_means()
