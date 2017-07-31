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
        x = np.array(x)
        self.n_data = x.shape[0]
        self.dim = x.shape[1]
        self._scale = np.mean(x, 0), np.std(x, 0)
        self.x = (x - self._scale[0]) / self._scale[1]    # data normalized
        if y is None:
            self.y = y
            self.types = set(y)
            self.n_types = len(self.types)
