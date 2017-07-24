"""
This file is deprecated. Please refer to BPNetwork instead
"""

# import numpy as np
# import pandas as pds
# from NetworkComponent import OutputNetworkLayer
#
#
# class BPPerceptron:
#     """
#     Single BPPerceptron, perceptron with only input and output layer
#     Notes: this is the interface with param check
#     """
#     def __init__(self, n_input, n_output, n_neurons):
#         # TODO: param check
#         self._n_input = n_input
#         self._n_output = n_output
#         self._layer = OutputNetworkLayer(n_priors=n_input, n_neurons=n_neurons)
#
#     def train(self, x, y):
#         x = np.array(x)
#         y = np.array(y)
#         assert x.shape == (self._n_input, ), \
#             "Train data x %s does not fit initialized shape %s" % (str(x.shape), self._n_input)
#         assert y.shape == (self._n_output, ), \
#             "Train data y %s does not fit initialized shape %s" % (str(y.shape), self._n_output)
#         self._layer.train(x, y)
#
#     def train_many(self, x, y):
#         x = np.array(x)
#         y = np.array(y)
#         assert x.shape[1] == self._n_input
#         assert y.shape[1] == self._n_output
#         assert x.shape[0] == y.shape[0]
#         for xrow, yrow in zip(x, y):
#             self._layer.train(xrow, yrow)
#
#     def predict(self, x):
#         x = np.array(x)
#         assert len(x.shape) == 1 and x.shape[0] == self._n_input
#         return self._layer.predict(x)
#
#     def get_weight(self):
#         return self._layer.get_weight()
#
#
# def test():
#     # not a successful test
#     d = pds.read_csv('Dataset/watermelon-tiny.csv')
#     d = d.sample(frac=1)
#
#     for col in d.columns:
#         c = d[col]
#         d[col] = (c - c.min()) / (c.max() - c.min())
#
#     train = d[d.columns[1:-1]]
#     truth = d[d.columns[-1]]
#     n = len(d)
#     m = int(n * 0.8)
#
#     b = BPPerceptron(8, 1, 1)
#     # b.train_many(train[:m], np.array(truth[:m]).reshape((-1, 1)))
#     for i in range(m):
#         b.train(np.array(train[i:i+1])[0], [truth[i:i+1].iloc[0]])
#         print(b.get_weight())
#     p = np.array([b.predict(x[1]) for x in train[m:].iterrows()])
#     print(p)
#     print(truth[m:])
#
#     # print(np.sum(np.abs(p - np.array(truth[m:])) < 0.5))
#
# # TODO: early stopping
# # TODO: regularization
#
# if __name__ == "__main__":
#     test()
