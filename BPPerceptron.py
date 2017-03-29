import numpy as np


class BPPerceptron:
    def __init__(self, n_input, n_hidden, n_output):
        # TODO: param check
        self._n = n_input
        self._n_output = n_output
        self._n_hidden = n_hidden
        self._hidden = [Neuron(n_input) for _ in range(n_hidden)]
        self._output = [Neuron(n_hidden) for _ in range(n_output)]

    def _predict_hidden(self, x):
        ans = [n.predict(x) for n in self._hidden]
        return np.array(ans)

    def _predict_output(self, y_hidden):
        ans = [n.predict(y_hidden) for n in self._output]
        return np.array(ans)

    def _update_output(self, prior_output, truth, output):
        g = [n.update(prior_output, truth, output) for n in self._output]
        return np.array(g)

    def _calc_hidden_gradient(self, neurons, output, subsequent_neurons):
        g = np.zeros(len(neurons))
        for i in range(len(neurons)):
            n = neurons[i]
            g[i] = np.sum([x.get_weight(i) * x.gradient for x in subsequent_neurons])
        g *= output * (1 - output)
        return g

    def _update_hidden(self, inputs, neurons, gradients):
        for n, g in zip(neurons, gradients):
            n.update(inputs, g=g)

    def feed(self, x, y):
        x = np.array(x)
        y = np.array(y)
        hidden_output = self._predict_hidden(x)
        final_output = self._predict_output(hidden_output)
        self._update_output(hidden_output, y, final_output)
        hidden_gradient = self._calc_hidden_gradient(self._hidden, hidden_output, self._output)
        self._update_hidden(x, self._hidden, hidden_gradient)

    def predict(self, x):
        pass

    def size(self):
        return {
            "input": self._n,
            "hidden": self._n_hidden,
            "output": self._n_output
        }

class NetworkLayer:
    def __init__(self, n_priors, n_neurons, ):

class Neuron:
    def __init__(self, n_priors, weight_init=None, activation_func=None, learn_rate=None):
        """
        Initialize Neuron with following param
        :param n_priors: int, number of prior level neurons
        :param weight_init: [float*n_input, threshold*1], random by default
        :param activation_func: lambda or function, Sigmoid by default 1/(1+exp(-x))
        """
        self._n = n_priors
        self._weight = weight_init or np.random.rand(n_priors + 1)
        self._activation_func = activation_func or (lambda x: 1/(1+np.exp(-x)))
        self._learn_rate = learn_rate or 0.01   # TODO: 0.01?
        self.gradient = None

    def predict(self, x):
        """
        Give x to be predicted on
        :param x: [float*n_input]
        :return: float, prediction
        """
        s = np.sum(np.array(list(x) + [-1]) * self._weight) # the last one is threshold
        y = self._activation_func(s)
        return y

    def update(self, x, y=None, y_=None, g=None):
        # TODO: other algorithm
        # TODO: nedd x?
        """
        Update neuron with BP algorithm
        :param x: [float*n_input]
        :param y: float, truth
        :param y_: float, prediction result
        :return:
        """
        if g is None:
            g = y_ * (1 - y_) * (y - y_)
        self._weight[:-1] += self._learn_rate * g * x
        self._weight[-1] += -self._learn_rate * g
        self.gradient = g
        return g

    def calc_gradient(self, y, y_):
        # TODO: bound to BP?
        """
        return output side gradient and hidden side gradient in BP
        :param y: float, truth
        :param y_: float, prediction
        :return:
        """
        g = y_ * (1 - y_) * (y - y_)
        self.gradient = g
        return g

    def set_learn_rate(self, rate=None):
        """
        If rate not specified, halving by default
        :param rate: float, new learning rate.
        :return: No return
        """
        self._learn_rate = rate or self._learn_rate * 0.5

    def set_weight(self, w):
        self._weight[:-1] = list(w)

    def set_threshold(self, t):
        self._weight[-1] = float(t)

    def get_weight(self, i=None):
        if i:
            return self._weight[i]
        else:
            return self._weight[:-1]

    def get_threshold(self):
        return self._weight[-1]




def test():
    b = BPPerceptron(2, 2, 1)
    b.feed([0, 0], [0])
    b.feed([0, 1], [1])
    b.feed([1, 0], [1])
    b.feed([1, 1], [0])



if __name__ == "__main__":
    test()