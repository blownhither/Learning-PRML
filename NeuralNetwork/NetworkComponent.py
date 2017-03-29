import abc
import numpy as np


class NetworkLayer(metaclass=abc.ABCMeta):
    """
    Abstraction of network layer containing some Neurons with similar settings
    No param check enforced in this class
    """
    def __init__(self, n_priors, n_neurons, weight_init=None, activation_func=None, learn_rate=None, next_layer=None):
        """
        :param n_priors: int, number of input this layer receives
        :param n_neurons: int, number of neurons in this layer
        :param weight_init: NOT IMPLEMENTED !
        :param activation_func: lambda or function, Sigmoid by default 1/(1+exp(-x))
        :param learn_rate: float, learning rate
        """
        # TODO: weight init
        self._neurons = [
            Neuron(n_priors, weight_init=None, activation_func=activation_func, learn_rate=learn_rate)
            for _ in range(n_neurons)
            ]
        self._n_prior = n_priors
        self._n_neurons = n_neurons
        self._activation_func = activation_func
        self._learn_rate = learn_rate
        self._next_layer = next_layer
        self._save_gradient = None

    def predict(self, x):
        """
        Produce layer output with input x
        :param x: [float] * n_priors
        :return: [float] * n_neurons
        """
        ans = [n.predict(x) for n in self._neurons]
        return np.array(ans)

    @abc.abstractmethod
    def gradient(self, y_, y=None):
        """
        :param y: truth
        :param y_: prediction
        :return:
        """
        pass

    def train(self, x, y, gradient=None):
        """
        Learn from input x and truth y_, following gradient (if not specified, will call self.gradient())
        :param x: [float] * n_priors, input x
        :param y: truth of y
        :param gradient: [float] * n_neurons, calculated if not specified
        :return: [float] * n_neurons, gradient for convenience
        """
        y_ = self.predict(x)
        gradient = gradient or self.gradient(y, y_)
        gradient = [n.train(x=x, y=y, y_=y_, g=g) for n, g in zip(self._neurons, gradient)]
        return gradient

    def set_next_layer(self, next_layer):
        self._next_layer = next_layer

    def get_next_layer(self):
        return self._next_layer

    def get_saved_gradient(self):
        return self._save_gradient

    def get_weight(self, i=None):
        """
        :param i: If i specified, get w_{i} from each neuron, or [w_{ix} for x in layer]
        :return:
        """
        ans = [n.get_weight(i) for n in self._neurons]
        return np.array(ans)

    def get_threshold(self):
        ans = [n.get_threshold() for n in self._neurons]
        return np.array(ans)

    def get_learn_rate(self):
        ans = [n.get_learn_rate() for n in self._neurons]
        return np.array(ans)

    def shape(self):
        return {
            "neurons": self._n_neurons,
            "prior": self._n_prior,
            "weight": self.get_weight(),
            "threshold": self.get_threshold(),
            "activation_function": self._activation_func,
            "learn_rate": self.get_learn_rate()
        }

    def size(self):
        return self._n_neurons


class OutputNetworkLayer(NetworkLayer):
    def __init__(self, n_priors, n_neurons, weight_init=None, activation_func=None, learn_rate=None):
        super(OutputNetworkLayer, self).__init__(n_priors, n_neurons, weight_init, activation_func, learn_rate)

    def gradient(self, y_, y=None):
        assert y_ is not None and y is not None
        self._save_gradient = y_ * (1 - y_) * (y - y_)
        return self._save_gradient


class HiddenNetworkLayer(NetworkLayer):
    def __init__(self, n_priors, n_neurons, weight_init=None, activation_func=None, learn_rate=None):
        super(HiddenNetworkLayer, self).__init__(n_priors, n_neurons, weight_init, activation_func, learn_rate)

    def gradient(self, y_, y=None):
        assert y is None
        l = self.get_next_layer()
        assert isinstance(l, NetworkLayer)
        g = [np.sum(l.get_weight(i) * l.get_saved_gradient()) for i in range(self._n_neurons)]
        g *= y_ * (1 - y_)
        return g


class Neuron:
    """
    Basic element in neural network, with some input weights, activation function and learning rate
    """
    def __init__(self, n_priors, weight_init=None, activation_func=None, learn_rate=None):
        """
        Initialize Neuron with following param
        :param n_priors: int, number of prior level neurons
        :param weight_init: [float*n_input, threshold*1], random by default
        :param activation_func: lambda or function, Sigmoid by default 1/(1+exp(-x))
        """
        self._n = n_priors
        self._weight = weight_init or np.random.rand(n_priors + 1)
        self._activation_func = activation_func or (lambda x: 1 / (1 + np.exp(-x)))
        self._learn_rate = learn_rate or 0.01
        self.gradient = None

    def predict(self, x):
        """
        Give x to be predicted on
        :param x: [float*n_input]
        :return: float, prediction
        """
        s = np.sum(np.append(np.array(x), [-1]) * self._weight)     # the last one is threshold
        y = self._activation_func(s)
        return y

    def train(self, x, y=None, y_=None, g=None):
        """
        Update neuron with BP algorithm (using gradient descent)
        :param x: [float] * n_input
        :param y: float, truth
        :param y_: float, prediction result
        :param g: gradient input
        :return: float, gradient before update
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

    def get_learn_rate(self):
        return self._learn_rate

    def get_weight(self, i=None):
        if i:
            return self._weight[i]
        else:
            return self._weight[:-1]

    def get_threshold(self):
        return self._weight[-1]
