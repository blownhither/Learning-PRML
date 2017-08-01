#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 9:10 PM
"""
import numpy as np


class GeneralClustering:
    def __init__(self, x, y=None, normalize=True):
        """
        General type for Clustering classes. Y is None for unsupervised clustering
        """
        x = np.array(x)
        self.size = x.shape[0]
        self.dim = x.shape[1]
        if normalize is True:
            self.x = (x - self._scale[0]) / self._scale[1]      # data normalized
            self._scale = np.mean(x, 0), np.std(x, 0)           # mean & std
        else:
            self._scale = None

        if y is not None:
            self.y = y
            self.types = set(y)
            self.n_types = len(self.types)

    @staticmethod
    def _external_index(x, y, y_):
        # s->same, d->different for each pair of samples
        # e.g. ss stands for pair with same label both in y and y_
        ss, sd, ds, dd = 0, 0, 0, 0

        x = np.array(x)
        n = len(x)
        for i in range(n):                  # for each pair
            for j in range(i + 1, n):
                if y[i] == y[j]:
                    if y_[i] == y_[j]:      # same-same
                        ss += 1
                    else:                   # same-diff
                        sd += 1
                else:
                    if y_[i] == y_[j]:      # diff-same
                        ds += 1
                    else:                   # diff-diff
                        ds += 1
        return ss, sd, ds, dd

    @staticmethod
    def jaccard_coefficient(x, y, y_):
        """
        return Jaccard Coefficient on arguments $\frac{SS}{SS + SD + DS}$
        :param x: Data
        :param y: Predicted label
        :param y_: Truth label
        :return: Jaccard Coefficient on arguments
        """
        ss, sd, ds, _ = GeneralClustering._external_index(x, y, y_)
        if ss + sd + ds == 0:
            return np.inf
        return float(ss) / (ss + sd + ds)

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
            return -1
        for i in range(n_):
            for j in range(i + 1, n_):
                sum_ += np.linalg.norm(cluster[i] - cluster[j])
        return 2 * sum_ / n_ / (n_ + 1)

    @staticmethod
    def _db_index(x, y):
        """
        Davies-Bouldin Index of a unsupervised clustering scheme
        :return: db_index
        """
        x = np.array(x)
        dim = x.shape[1]
        types = list(set(y))
        k = len(types)
        avg = np.zeros(k)
        centers = np.zeros((k, dim))
        for idx, t in enumerate(types):
            cluster = x[y == t]
            avg[idx] = GeneralClustering._cluster_avg(cluster)
            centers[idx] = np.mean(cluster, 0)

        dbi = -np.ones((k, k))                # diagonal ignored
        for i in range(k):
            for j in range(i + 1, k):
                temp = (avg[i] + avg[j]) / np.linalg.norm(centers[i] - centers[j])
                dbi[i, j] = temp                        # symmetrical mat
                dbi[j, i] = temp
        dbi = np.max(dbi, 1)
        return float(np.sum(dbi) / k)
