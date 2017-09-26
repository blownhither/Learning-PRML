#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 8:53 PM
"""
import numpy as np
from Clustering.GeneralClustering import GeneralClustering


class LVQ(GeneralClustering):
    """
    Learning Vector Quantization
    """
    def __init__(self, x, y):
        super(LVQ, self).__init__(x, y)
        self.centers = None         # center of each cluster (prototype vector)

    def fit(self, learning_rate, max_iter=1000):
        self.centers = np.random.random((self.n_types, self.dim))
        for iter_ in range(max_iter):
            choice = np.random.randint(self.size)
            x_, y_ = self.x[choice], self.y[choice]             # random sample
            dist = np.linalg.norm(x_ - self.centers, 2, 1)      # distance to each prototype vectors
            idx = np.argmin(dist)                               # choose nearest prototype vector
            if y_ == idx:
                update = (1 - learning_rate) * self.centers[idx] + learning_rate * x_
            else:
                update = (1 + learning_rate) * self.centers[idx] - learning_rate * x_
            self.centers[idx] = update

            if iter_ % 50 == 0:
                n_sample = max(int(self.size * 0.05), 1)
                idx = np.random.randint(0, self.size, n_sample)
                samples = self.x[idx]                           # 5% samples
                prediction = self.predict(samples)
                jc = self.jaccard_coefficient(samples, prediction, self.y[idx])
                print('Iter %d, Jaccard Coefficient %g' % (iter_, jc))

    def _predict_one(self, x):
        return np.argmin(np.linalg.norm(x - self.centers, 2, 1))    # nearest prototype vector

    def predict(self, x):
        x_ = np.array(x)
        if x_.ndim == 1:
            return self._predict_one(x)
        elif x_.ndim == 2:
            return np.array([self._predict_one(d) for d in x_])
        else:
            raise TypeError('predict(x) takes row-wise matrix x or single-row')


def _test():
    import pandas as pds
    data = pds.read_csv('../Dataset/watermelon-tiny.csv')
    x = np.array(data[data.columns[1:-1]]).repeat(20, axis=0)
    y = np.array(data[data.columns[-1]]).repeat(20)
    lvq = LVQ(x, y)
    lvq.fit(1e-2, 100000)


if __name__ == '__main__':
    _test()
