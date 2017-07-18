#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence 
@time: 7/17/17 9:50 PM
"""
from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self):
        self.x = None               # train set
        self.y = None
        self.validation_x = None    # validation set
        self.validation_y = None
        self.dim = None             # shape info
        self.size = None
        self.head = None            # tree head

    def fit(self, x, y):
        """
        Fit a decision tree with given x and y
        :param x: row-wise discrete data
        :param y: class label
        """
        x = np.array(x)
        y = np.array(y)
        index = np.arange(len(x))   # shuffle and split validation set
        np.random.shuffle(index)
        split = np.floor(len(index) * 0.95)
        self.x = x[index[:split]]
        self.y = y[index[:split]]
        self.validation_x = x[index[:split]]
        self.validation_y = y[index[:split]]
        self.dim = self.x.shape[1]  # shape information
        self.size = self.x.shape[0]

        data_idx = np.arange(self.size)
        dim_idx = np.arange(self.dim)
        self.head = self._generate(data_idx, dim_idx)

    def _generate(self, data_idx, dim_idx):
        """
        generate (sub)tree
        :param data_mask: maks of dataset, try to save space
        :param dim_mask: mask of dim
        :return:
        """
        # case 1
        case1 = True
        labels = self.y[data_idx]
        temp = labels[0]
        for l in labels:
            if l != temp:
                case1 = False
                break
        if case1 is True:
            node = DecisionTreeNode()
            node.leaf = True
            node.label = temp
            return node

        # case 2
        if len(dim_idx) == 0 or self._same_attribute_value(data_idx, dim_idx):
            node = DecisionTreeNode()
            node.leaf = True
            node.label = self._most_frequent(labels)
            return node

        # case 3
        div_dim = self._best_division(data_idx, dim_idx)




    def _most_frequent(self, labels):
        return np.argmax(np.bincount(labels))

    def _same_attribute_value(self, data_idx, dim_idx):
        for col in dim_idx:
            temp = self.x[data_idx[0], col]
            for row in data_idx:
                if self.x[row, col] != temp:
                    return False
        return True

    def _best_division(self, data_idx, dim_idx):
        """
        Use Gini index to choose one best dimension to split on
        :return: dimension index
        """
        arg = np.argmax([self._gini_index(data_idx, d) for d in dim_idx])[-1]
        return dim_idx[arg]

    def _gini_index(self, data_idx, dim):
        counter = Counter(self.x[data_idx, dim])


class DecisionTreeNode:
    def __init__(self):
        self.dim = None         # divide on a dim if not leaf
        self.leaf = False       # whether node is leaf
        self.label = None       # label can be decided if node is leaf
        self.children = None    # a dict mapping attribute value to children


if __name__ == '__main__':
    pass