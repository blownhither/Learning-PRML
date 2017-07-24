#!/usr/bin/env python2
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
        :param y: class label, ranging from 0...n
        """
        x = np.array(x)
        y = np.array(y)
        index = np.arange(len(x), dtype=np.int)     # shuffle and split validation set
        np.random.shuffle(index)
        split = np.int(np.floor(len(index) * 0.95))
        self.x = x[index[:split]]
        self.y = y[index[:split]]
        self.validation_x = x[index[:split]]
        self.validation_y = y[index[:split]]
        self.dim = self.x.shape[1]                  # shape information
        self.size = self.x.shape[0]

        data_idx = np.arange(self.size, dtype=np.int)             # start generating decision tree
        dim_idx = np.arange(self.dim, dtype=np.int)
        self.head = self._generate(data_idx, dim_idx)

    def _generate(self, data_idx, dim_idx):
        """
        generate (sub)tree
        :param data_idx: index of dataset, try to save space
        :param dim_mask: mask of dim
        :return:
        """
        # case 1, same label for all samples
        case1 = True
        labels = self.y[data_idx]
        temp = labels[0]
        node = DecisionTreeNode()                   # new node to be returned
        for l in labels:
            if l != temp:
                case1 = False
                break
        if case1 is True:
            node.leaf = True                        # new node is a leaf
            node.label = temp                       # assign new node to the only label
            return node

        # case 2, same value for all attributes
        if len(dim_idx) == 0 or self._same_attribute_value(data_idx, dim_idx):
            node.leaf = True                            # new node is leaf
            node.label = self._most_frequent(labels)    # assign new node to the most frequent label
            return node

        # case 3, split on one optimal dimension to generate subtree
        div_dim = self._best_division(data_idx, dim_idx, labels)
        data = self.x[data_idx, div_dim]
        node.children = []
        node.attribute_value = []
        node.leaf = False                               # new node is not leaf
        node.dim = div_dim                              # split on optimal dim
        for value in set(data):
            # TODO: pre-stored data col values should improve efficiency
            new_data_idx = data_idx[np.where(value == data)]
            # new_labels = self.y[new_data_idx]
            new_dim_idx = [x for x in dim_idx if x != div_dim]
            new_node = self._generate(new_data_idx, new_dim_idx)
            node.children.append(new_node)
            node.attribute_value.append(value)
        if not len(node.children) > 1:
            self._same_attribute_value(data_idx, dim_idx)
        return node

    @staticmethod
    def _most_frequent(labels):
        return np.argmax(np.bincount(labels))

    def _same_attribute_value(self, data_idx, dim_idx):
        for col in dim_idx:
            temp = self.x[data_idx[0], col]
            for row in data_idx:
                if self.x[row, col] != temp:
                    return False
        return True

    def _best_division(self, data_idx, dim_idx, labels):
        """
        Use Gini index to choose one best dimension to split on
        :return: dimension index
        """
        arg = np.argmin([self._gini_index(data_idx, d, labels) for d in dim_idx])
        return dim_idx[arg]

    def _gini_index(self, data_idx, dim, labels):
        n = len(data_idx)
        data = self.x[data_idx, dim]
        counter = Counter(data)
        ans = 0.0
        for key, value in counter.iteritems():
            ans += self._gini(labels[data == key]) * value
        return ans / n

    @staticmethod
    def _gini(x):
        return 1 - np.linalg.norm(np.array(Counter(x).values(), dtype=np.float)) / float(len(x) ** 2)

    def print_tree(self):
        stack = [self.head]
        while stack:
            temp = []
            s = ""
            for node in stack:
                s += str(node)
                if node.children:
                    temp.extend(node.children)
            print(s)
            stack = temp


class DecisionTreeNode:
    def __init__(self):
        self.dim = None         # divide on a dim if not leaf
        self.leaf = False       # whether node is leaf
        self.label = None       # label can be decided if node is leaf
        self.children = None    # a dict mapping attribute value to children
        self.attribute_value = None # attribute value for each children

    def __str__(self):
        if self.leaf is True:
            return 'Label: %d ' % self.label
        else:
            s = 'Dim: %d\n' % self.dim
            for v in self.attribute_value:
                s += str(v) + ' '
            return s


def test_dt():
    dt = DecisionTree()
    import pandas as pds
    df = pds.read_csv('Dataset/watermelon-tiny.csv')
    x = np.array(df[df.columns[1:-3]])
    y = np.array(df[df.columns[-1]])
    dt.fit(x, y)
    # dt.print_tree()

    # to use plot_tree you need to install graphviz and pygraphviz
    from Util.plot_tree import plot_tree
    plot_tree(dt.head, 'tmp/tree.png')


if __name__ == '__main__':
    test_dt()