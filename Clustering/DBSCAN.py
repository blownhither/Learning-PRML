#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 9:10 PM
"""
import numpy as np
import random
import warnings

from Clustering.GeneralClustering import GeneralClustering

# TODO: use kd-tree


class DBSCAN(GeneralClustering):
    """ Density-Based Spatial Clustering of Applications with Noise, an unsupervised clustering method """
    def __init__(self, x):
        super(DBSCAN, self).__init__(x)
        self.labels = None

    def fit(self, epsilon, min_points):
        """
        Fit DBSCAN on dataset x
        :param epsilon: neighborhood radius (spatial distance)
        :param min_points: minimal size min_points
        :return: labels assignment, [0:k, -1 for noise]
        """
        cores = set()
        for idx, sample in enumerate(self.x):
            if len(self._neighbors(sample, epsilon)) - 1 >= min_points:  # don't count sample itself
                cores.add(idx)
        next_label = 0
        labels = -np.ones(self.size)
        unvisited = set(range(self.size))                               # TODO: use array
        if len(cores) <= 1:
            warnings.warn('Invalid number of cores found (0 or 1), consider change parameters')
        while len(cores) > 0:
            choice = random.sample(cores, 1)[0]
            queue_ = [choice]
            unvisited.remove(choice)
            labels[choice] = next_label                                 # a new cluster
            while len(queue_) > 0:
                q = queue_.pop(0)
                neighbors = self._neighbors(self.x[q], epsilon)
                if len(neighbors) - 1 >= min_points:                    # don't count q itself
                    for n in neighbors:
                        if n not in unvisited:                          # only on unvisited
                            continue
                        queue_.append(n)
                        unvisited.remove(n)
                        cores.discard(n)                                # possible remove
                        labels[n] = next_label                          # belong to the new cluster
            cores.remove(choice)
            next_label += 1
        self.labels = labels.copy()
        self._plot(self.x, labels)
        return labels

    def _neighbors(self, sample, epsilon):
        """ All neighbors index within distance of epsilon from sample """
        dist = np.linalg.norm(sample - self.x, 2, 1)
        return np.where(dist < epsilon)[0]

    def get_labels(self):
        return self.labels.copy()

    def test(self):
        dbi = self._db_index(self.x, self.labels)                   # include noise, TBD
        return dbi


def _test():
    import pandas as pds
    df = pds.read_csv('../Dataset/watermelon-numeric.csv')
    d = DBSCAN(df)
    labels = d.fit(0.75, 5)
    print(labels)
    dbi = d.test()
    print(dbi)


if __name__ == '__main__':
    _test()
