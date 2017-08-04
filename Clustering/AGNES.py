#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 9:10 PM
"""
import numpy as np

from Clustering.GeneralClustering import GeneralClustering


class AGNES(GeneralClustering):
    """ Agglomerative Nesting algorithm for unsupervised clustering """
    def __init__(self, data):
        super(AGNES, self).__init__(data)

    def fit(self, k):
        """
        Fit with a clustering forest until it contains k clusters
        Using max pair distance as distance between two clusters
        :param k: Final number of clusters
        :return: labels assigned to each sample
        """
        dist = np.zeros((self.size, self.size))
        labels = np.arange(self.size)
        for i in range(self.size):
            dist[i, i] = np.inf
            for j in range(i + 1, self.size):
                temp = np.linalg.norm(self.x[i] - self.x[j])
                dist[i, j] = temp
                dist[j, i] = temp
        for _ in range(self.size, k, -1):
            i, j = np.unravel_index(dist.argmin(), dist.shape)  # nearest cluster pair
            labels[labels == j] = i                             # combine cluster i and j
            temp = np.maximum(dist[i, :], dist[j, :])           # re-calculate dist mat
            dist[i, :] = temp
            dist[:, i] = temp
            dist[j, :] = np.inf                                 # cluster j disappear
            dist[:, j] = np.inf

        # rearrange labels
        for i, v in enumerate(set(labels)):
            if i != v:
                labels[labels == v] = i
        return labels


def _test():
    import pandas as pds
    data = pds.read_csv('../Dataset/watermelon-numeric.csv')
    a = AGNES(data)
    labels = a.fit(7)
    print(labels)
    for i in range(7):
        print(np.where(labels == i))


if __name__ == '__main__':
    _test()
