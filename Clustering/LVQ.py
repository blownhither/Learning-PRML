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

    def fit(self, learning_rate):
        self.centers = np.random.random((self.n_types, self.dim))
        while True:
            choice = np.random.randint(self.n_data)
            x_, y_ = self.x[choice], self.y[choice]             # random sample
            dist = np.linalg.norm(x_ - self.centers, 2, 1)      # distance to each prototype vectors
            idx = np.argmin(dist)                               # choose nearest prototype vector
            if y_ == idx:
                update = (1 - learning_rate) * self.centers[idx] + learning_rate * x_
            else:
                update = (1 + learning_rate) * self.centers[idx] - learning_rate * x_
            self.centers[idx] = update



if __name__ == '__main__':
    pass