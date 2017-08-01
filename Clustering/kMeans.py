#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/27/17 8:45 PM
"""
import numpy as np

from Clustering.GeneralClustering import GeneralClustering


class KMeans(GeneralClustering):
    """
    Simple k-means, using square distance
    """
    def __init__(self, x):
        """
        Feed data matrix
        :param x:
        :param k: number of cluster expected
        """
        super(KMeans, self).__init__(x)
        self.centers = None                                     # center of each cluster
        self.labels = None                                      # cluster id for each sample
        self.k = None                                           # number of clusters
        self.colors = None                                      # preset color map helping plotting
        self._pca = None                                        # PCA object helping plotting

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
        if plot:                                        # if plot, setup colors and PCA
            from matplotlib import pyplot as plt, cm    # import here in case not needed
            from sklearn.decomposition import PCA
            plt.ion()                                   # non-blocking mode when figure is plotted
            self.colors = cm.rainbow(np.linspace(0, 0.70, self.k))
            self._pca = PCA(2)                          # help reduce dim to 2 so as to be plotted
            self._pca.fit(self.x)

        for iter_ in range(max_iter):
            self.labels = self._iter()                  # newly assigned labels
            for i in range(k):
                cluster = self.x[self.labels == i]
                if len(cluster) > 0:                    # if empty cluster, keep original center
                    self.centers[i] = np.mean(cluster, 0)

            if iter_ % 5 == 0:
                dbi = self._db_index(self.x, self.labels)   # evaluate un-supervised clustering
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
        dist = np.zeros((self.size, self.k))          # distance matrix
        for i in range(self.k):
            dist[:, i] = np.linalg.norm(self.x - self.centers[i], 2, 1)
        return np.argmin(dist, 1)                       # assign to nearest cluster

    def _plot(self):
        """
        Plot scatter points with different colors
        """
        from matplotlib import pyplot as plt
        if self.dim == 1:
            raise NotImplementedError("plotting Dim = 1 not implemented")
        for i in range(self.k):
            points = self.x[self.labels == i]               # points in a cluster
            if len(points) == 0:
                continue
            if self.dim > 2:
                points = self._pca.transform(points)        # transform to 2-d points
            plt.scatter(points[:, 0], points[:, 1], marker='o', color=self.colors[i], alpha=.6)
            pca_center = self._pca.transform(self.centers[i].reshape((1, -1)))[0]
            plt.scatter(pca_center[0], pca_center[1], 100, marker='+', color='red') # draw center point
        plt.show()
        plt.pause(0.1)                                      # pause long enough to be seen
        plt.clf()                                           # clear graph

    def get_centers(self):
        return (self.centers * self._scale[1]) + self._scale[0] # de-normalization

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
    import pandas as pds
    data = pds.read_csv('../Dataset/watermelon-tiny.csv')
    k = KMeans(data[['Density', 'Sugar']])
    k.fit(3, plot=True)
    input()

if __name__ == '__main__':
    _test_k_means()
