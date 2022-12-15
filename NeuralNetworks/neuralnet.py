import numpy as np
from sklearn.utils import shuffle


# ENTIRE FILE IS FUBAR
# HAVING A NEURON CLASS SENT ME IN THE WRONG DIRECTION


def append_b(matrix):
    m = len(matrix)
    bias_column = np.ones((m, 1))
    return np.concatenate((bias_column, matrix), axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:

    def __init__(self, weights, is_bias=False):
        self.is_bias = is_bias

        # each neuron will hold the weights of the edges that feed into it
        # and the partial gradients with respect to those weights
        self.w = weights
        self.w_gradient = []

        # assigned in the neural net
        self.partial_L = 0

        self.value = None

    def evaluate(self, input_arr):
        if self.is_bias:
            return 1

        dot_product = np.dot(self.w, input_arr)
        self.value = sigmoid(dot_product)
        return self.value

    def get_sigma_deriv(self):
        if self.is_bias:
            return None

        return self.value * (1 - self.value)


class NeuronLayer:

    def __init__(self, width, num_inputs, initialize_zero=True):
        self.width = width
        self.num_inputs = num_inputs
        self.output = None
        self.input = None

        self.neurons = []

        if initialize_zero:
            # initialize weights as 0
            init_array = np.zeros(num_inputs)
        else:
            # initialize weights as 1
            init_array = np.array([1] * num_inputs)

        for i in range(width):
            # first neuron in layer is bias term
            if i == 0 and width > 1:
                neuron = Neuron(init_array, is_bias=True)
            else:
                neuron = Neuron(init_array)

            self.neurons.append(neuron)

    def evaluate(self, input_arr):
        self.input = input_arr

        output = []
        for neuron in self.neurons:
            output.append(neuron.evaluate(input_arr))

        self.output = np.array(output)
        return self.output





class NeuralNet:

    def __init__(self, num_layers, width, gamma_0, max_epoch, init_zero=True):
        # num layers doesnt include imput layer
        # ie num_layers = 3 -> 2 hidden layers and output layer
        self.num_layers = num_layers
        self.width = width
        self.gamma_0 = gamma_0
        self.max_epoch = max_epoch
        self.init_zero = init_zero

        self.layers = []

        self.x = None
        self.y = None

    def train(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

        # augments x
        x_b = append_b(x_train)
        m, n = x_b.shape

        # build the neural net once we know the dim of x
        self.build_net(n)

        for t in range(self.max_epoch):
            x_shuffle, y_shuffle = shuffle(x_b, y_train)

    def forward_propagation(self, x):

        rolling_input = x
        for layer in self.layers:
            rolling_input = layer.evaluate(rolling_input)
            print(rolling_input)
            if layer.width == 1:
                return rolling_input

    def back_propagation(self, x_i, y, y_star):


        for i in range(self.num_layers):
            # go through layers backyards
            layer_idx = self.num_layers - 1 - i
            # from notation
            h = layer_idx + 1

            current_layer = self.layers[layer_idx]

            # set the partial derivitve of loss with respect to y in the output neuron
            if i == 0:
                # current_layer = output layer
                output_neuron = current_layer.neurons[0]
                output_neuron.partial_L = y - y_star

            for j in range(current_layer.width):
                layer_inputs = current_layer.inputs
                neuron_partial = self.neuron_partial_L(h, j)








    def get_neuron(self, h, i):
        """gets the ith neuron in layer h"""
        layer_index = h - 1
        layer = self.layers[layer_index]
        return layer.neurons[i]

    def neuron_partial_L(self, h, i):
        """ computes dL / dz(h, i), h is layer, i is neuron index in layer h"""
        if h == self.num_layers:
            output_neuron = self.get_neuron(h, i)
            return output_neuron.partial_L


        depth = self.num_layers - h

    def build_net(self, num_inputs):
        """num_inputs = dim(x), builds a layered neural net to match data"""
        for i in range(self.num_layers):

            if i == 0:
                # first layer will have width = self.width, num_imputs = dim(x)
                layer = NeuronLayer(self.width, num_inputs)
            elif i == self.num_layers - 1:
                # last layer will have width = 1, and num_inputs = self.width
                layer = NeuronLayer(1, self.width)
            else:
                # intermediary layers will have width = num_inputs = self.width
                layer = NeuronLayer(self.width, self.width)

            self.layers.append(layer)
