import sys
from collections import Counter

import numpy as np


class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        self._x = self._y = None    # Mat-like, data fed
        self._prior = None          # [int], prior probability distribution
        # self._cond = None           # [[]], conditional probability distribution
        # self._x_types = None        # {key:set}, distinct types for each feature
        self._y_types = None        # [], all possible values for each element in y
        self._x_ntypes = None       # [int], counts of all possible values for each element in x
        self._y_ntypes = None       # int, counts for all possible values for each element in y
        self._n_samples = self._n_features = None
        self.smoothing = smoothing  # Smoothing coefficient in Bayes estimation

    def feed(self, x, y, x_types=None, y_types=0):
        """
        feed data into Naive Bayes Classifier, typically X[m*n], Y[n]
        :param x: Matrix with each column representing X of a sample
        :param y: 1-D Array representing Y of a sample, corresponding to each in X
        :param x_types:
        :param y_types:
        :return: Nothing returned
        """
        try:
            # TODO: param check
            self._prior = Counter(y)
            self._y_types = list(self._prior.keys())
            self._y_ntypes = max(len(self._y_types), y_types)
            self._x = np.array(x)
            x = self._x
            self._y = np.array(y)
            self._n_features, self._n_samples = x.shape
        except Exception as e:
            sys.stderr.write("Unexpected Parameters in NaiveBayesClassifier.feed")
            raise e
        self._x_ntypes = [len(set(row)) for row in x]

        # self._cond = dict([(self._y_types[i], Counter()) for i in range(self._y_ntypes)])
        # for i in range(self._n_samples):
        #     self._cond[y[i]] += Counter(x[:, i])

        print("Done forming distribution. ")

    def _predict(self, x, smoothing=True):
        """
                Predict the distirbution of y with given x, return array.
                :param x: []
                :return: {y_type: float}, possibilities of each
                """
        ans = {}
        if smoothing:
            sm = self.smoothing
        else:
            sm = 0
        for y in self._y_types:
            c_y = self._prior[y]
            prod = [0] * self._n_features
            for f in range(self._n_features):
                count = np.count_nonzero(np.logical_and(self._x[f] == x[f], self._y == y))
                prod[f] = (count + sm) / (c_y + sm * self._x_ntypes[f])
            prod = np.prod(prod)

            prior = (c_y + sm) / (self._n_samples + sm * self._y_ntypes)
            ans[y] = prior * prod
        return ans

    def predict_one(self, x, smoothing=True):
        """
        Predict the distrIbution of y with given x, return array.
        :param x: []
        :param smoothing: T/F
        :return: {y_type: float}, possibilities of each
        """
        ans = self._predict(x, smoothing)
        s = np.sum(list(ans.values()))
        for k in ans.keys():
            ans[k] /= s
        return ans

    def predict(self, x, smoothing=True):
        _, n_samples = x.shape
        ans = [None] * n_samples
        for i in range(n_samples):
            d = self._predict(x[:, i], smoothing)
            m = -1
            km = None
            for k, v in d.items():
                if v > m:
                    km = k
                    m = v
            ans[i] = km
        return ans


def test():
    x = np.array(np.mat("1 1 1 1 1 2 2 2 2 2 3 3 3 3 3; 1 2 2 1 1 1 2 2 3 3 3 2 2 3 3"))
    y = np.array(np.mat("1 1 2 2 1 1 1 2 2 2 2 2 2 2 1"))[0].tolist()
    nbc = NaiveBayesClassifier()
    nbc.feed(x, y)
    ans = nbc.predict_one(x=[2, 1])
    print(ans)

    ans = nbc.predict(x=np.array([[2], [1]]))
    print(ans)

if __name__ == '__main__':
    test()