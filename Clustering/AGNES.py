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
        Fit a clustering forest that contains k trees
        :param k: Final number of clusters
        :return:
        """
        

