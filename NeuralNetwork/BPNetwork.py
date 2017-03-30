from NeuralNetwork.NetworkComponent import OutputNetworkLayer, HiddenNetworkLayer, NetworkLayer
import numpy as np
import pandas as pds

class BPNetwork():
    def __init__(self, n_input, n_output, hidden, learn_rate=0.01):
        """
        Initialize a NN using BP algorithm
        :param n_input: int, the number of inputs
        :param n_output:int, the number of outputs
        :param hidden: [int], specify number of neurons in each hidden layer
        """
        self._n_input = n_input
        self._n_output = n_output
        self._hidden = hidden or []
        self._layers = None
        self._learn_rate = learn_rate
        self._build()

    def _build(self):
        self._layers = [None] * (len(self._hidden) + 1)
        n_priors = self._n_input
        for i in range(len(self._hidden)):
            n = self._hidden[i]
            self._layers[i] = HiddenNetworkLayer(n_priors=n_priors, n_neurons=n, learn_rate=self._learn_rate)
            n_priors = n
        self._layers[-1] = OutputNetworkLayer(n_priors=n_priors, n_neurons=self._n_output, learn_rate=self._learn_rate)
        for i in range(len(self._layers) - 1):
            self._layers[i].set_next_layer(self._layers[i+1])

    def predict(self, x):
        x = np.array(x)
        assert len(x) == self._n_input
        for layer in self._layers:
            x = layer.predict(x)
        return x

    def train(self, x, y):
        x = np.array(x)
        n = len(self._layers)
        result = [None] * (n + 1)
        result[0] = x
        for i in range(n):
            x = self._layers[i].predict(x)
            result[i + 1] = x
        self._layers[-1].update(x=result[-2], y_=result[-1], y=y)     # output is special
        # self._layers[-1].gradient(y_=result[-1], y=y)
        for i in range(n-2, -1, -1):
            l = self._layers[i]
            # l.gradient(y_=result[i+1])
            l.update(result[i], result[i+1])
        return

    def train_many(self, x, y):
        x = np.array(x)
        y = np.array(y)
        assert x.shape[1] == self._n_input
        assert y.shape[1] == self._n_output
        for row in range(len(x)):
            self.train(x[row], y[row])

    def get_layer(self, i=None):
        if i is None:
            return self._layers
        else:
            return self._layers[i]


def test():
    d = pds.read_csv('Dataset/watermelon-tiny.csv')
    d = d.sample(frac=1)

    for col in d.columns:
        c = d[col]
        d[col] = (c - c.min()) / (c.max() - c.min())

    train = d[d.columns[1:-1]]
    truth = d[d.columns[-1]]
    truth = pds.DataFrame({
        0: truth == 0,
        1: truth == 1
    }).astype(np.int)

    n = len(d)
    m = int(n * 0.8)

    b = BPNetwork(8, 2, [20,20,20], 0.5)
    for _ in range(100):
        b.train_many(train, truth)


    p = np.array([b.predict(x[1]) for x in train[m:].iterrows()])
    print("My prediction:")
    print((p + 0.5).astype(np.int))
    print("Answer:")
    print(truth[m:])


if __name__ == "__main__":
    test()