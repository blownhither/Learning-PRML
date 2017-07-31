#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 9:10 PM
"""
import numpy as np


class GeneralClustering:
    def __init__(self, x, y=None):
        """
        General type for Clustering classes. Y is None for unsupervised clustering
        """
        x = np.array(x)
        self.n_data = x.shape[0]
        self.dim = x.shape[1]
        self._scale = np.mean(x, 0), np.std(x, 0)
        self.x = (x - self._scale[0]) / self._scale[1]    # data normalized
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
