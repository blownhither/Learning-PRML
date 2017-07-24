#!/usr/bin/env python3
# encoding: utf-8

"""
@author: blownhither@github
@license: Apache Licence 
@time: 7/24/17 11:16 PM
"""
import pygraphviz as pgv

""" Require: graphviz, pygraphviz """


def plot_tree(tree, filename):
    """
    Plot a tree with pygraphviz. Node attributes specification see below
    :param tree: having attribute 'children' and '__str__'
    :param filename: filename (*.png) to save the plotted graph
    """
    assert hasattr(tree, 'children')
    assert hasattr(tree, '__str__')
    prefix = '0'
    g = pgv.AGraph()
    g.add_node('0', label=str(tree))
    _rec_plot_tree(tree, prefix, g)
    g.layout('dot')
    g.draw(filename)


def _rec_plot_tree(head, prefix, graph):
    if head.children is None:
        return
    for i, v in enumerate(head.children):
        name = prefix + str(i)
        graph.add_node(name, label=str(v))
        graph.add_edge(prefix, name)
        _rec_plot_tree(v, name, graph)


def _test_plot_tree():
    class Node:
        _id = 0

        def __init__(self):
            self.id = Node._id
            self.children = []
            Node._id += 1

        def __str__(self):
            return str(self.id)

    l = [Node() for _ in range(4)]
    l[0].children.append(l[1])
    l[0].children.append(l[2])
    l[2].children.append(l[3])
    plot_tree(l[0], '../tmp/gviz.png')


if __name__ == '__main__':
    _test_plot_tree()