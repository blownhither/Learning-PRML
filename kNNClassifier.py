import numpy as np


class kNNClassifier:
    def __init__(self):
        self.x = None
        self.y = None
        self.k = None

        self.n_features = None
        self.n_samples = None

    def train(self, x, y):
        x = np.array(x)
        y = np.array(y)

        assert x.ndim == 2
        self.n_samples, self.n_features = x.shape
        assert len(y) == self.n_samples

        # lazy
        self.x = x
        self.y = y

    def _predict_one(self, x):
        dist = np.array([np.linalg.norm(v - x) for v in self.x])
        index = dist.argsort()
        neighbors = self.y[index[:self.k]]
        return np.sum(neighbors, 0)

    def predict(self, x, k):
        self.k = k
        pred = np.array([self._predict_one(v) for v in self.x])
        return pred / self.k


def test():
    import pandas as pds
    df = pds.read_csv('Dataset/watermelon-tiny.csv')
    index = np.arange(len(df))
    np.random.shuffle(index)
    df = df.iloc[index]
    x = df[df.columns[1:-1]]
    y = df[df.columns[-1]] - 1

    l = kNNClassifier()
    # a = np.arange(10).reshape((-1, 1))
    # b = np.arange(10)
    # l.train(a, b)
    l.train(x, y)

    pred = l.predict(x, 7)
    pred = (pred > 0.5)

    print(y)
    print(pred)

    accu = np.count_nonzero(pred == np.array(y)) / float(len(pred))
    print(accu)

if __name__ == '__main__':
    test()

