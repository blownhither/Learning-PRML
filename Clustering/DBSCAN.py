#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence
@time: 7/31/17 9:10 PM
"""
import numpy as np

from Clustering import GeneralClustering


class DBSCAN(GeneralClustering):
    def __init__(self, x):
        super(DBSCAN, self).__init__(x)
        self.labels = None

    def fit(self, epsilon, min_points):
        """
        Fit DBSCAN on dataset x
        :param epsilon: neighborhood radius (spatial distance)
        :param min_points: minimal size min_points
        :return: labels assignment
        """
        cores = set()
        

