import numpy as np
from sklearn.utils import shuffle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1.0 - sigmoid_x)

class Layer:

    def __init__(self, num_inputs, num_outputs, init_zero=False, is_output=False):

        self.num_inputs = num_inputs
        # unless is_output is True, assume one of the "outputs" is the bias term
        self.num_outputs = num_outputs
        self.is_output = is_output

        if is_output:
            # need a set of of weights for all nodes on output layer
            self.num_activated_outputs = num_outputs
        else:
            # need one less set for not ouptut layers due to bias
            self.num_activated_outputs = num_outputs - 1

        if init_zero:
            self.weights = np.zeros((num_inputs, self.num_activated_outputs))
        else:
            self.weights = np.random.rand(num_inputs, self.num_activated_outputs)

        # list of activations for layer
        self.output = np.zeros((1, num_outputs))
        # bias term
        if num_outputs > 1 and not is_output:
            self.output[0][0] = 1

        # track last inputs received by layer
        self.inputs = None

        # list of derivitives for layer
        self.partials = np.zeros(self.weights.shape)

    def activate_layer(self, layer_input):
        self.inputs = layer_input.reshape((1, -1))

        output_shape = self.output.shape

        if self.is_output:
            s_func_evals = np.dot(self.inputs, self.weights)
            # in lecture notes y = <w> dot <input vec> and not sigmoid
            activation = sigmoid(s_func_evals)
            self.output = activation.reshape(output_shape)
        else:
            # first activation is one for all layers with bias term
            bias_term = np.array([1])
            s_func_evals = np.dot(self.inputs, self.weights)
            # len activations = len num outputs -1
            activations = sigmoid(s_func_evals)
            self.output = np.concatenate((bias_term, activations[0])).reshape(output_shape)
        return self.output


class NeuralNet:

    def __init__(self, net_structure=(5, 3, 3, 1), init_zero=False):
        """

        :param net_structure:
                indicates the network structure. First number is length of input vectrors, following numbers are widths
                of each hidden layer, final number is number of network outputs.
        :param init_zero:
                When false, network weights will be random, when true they will all init at 0. Primariliy for test
                purposes
        """
        # num layers doesnt include input layer
        # ie num_layers = 3 -> 2 hidden layers and output layer
        self.net_structure = net_structure

        self.layers = []

        self.gamma_0 = None

        self.x = None
        self.y = None

        self.training_y_hat = []

        self.build_net(net_structure, init_zero)

    def predict(self, new_x):
        predictions = []
        for x in new_x:
            propagate = self.forward_propagate(x).flatten()[0]
            pred = 1 if propagate > .5 else 0
            predictions.append(pred)
        return np.array(predictions)


    def train(self, x_train, y_train, max_epoch=100, learning_rate=0.05, d=1):
        m, n = np.shape(x_train)

        if self.net_structure[0] != n:
            print("data shape does not match net structure")
            return

        self.gamma_0 = learning_rate
        for epoch in range(max_epoch):

            gamma_t = self.gamma_schedule(d=d, t=epoch)
            epoch_sum_square_error = 0

            x, y = shuffle(x_train, y_train)

            for i, x_i in enumerate(x):
                # ground truth
                y_i = y[i]
                # networks guess
                y_hat = self.forward_propagate(x_i)

                if epoch == max_epoch -1:
                    self.training_y_hat.append(y_hat)

                error = y_hat - y_i
                epoch_sum_square_error += self.square_error(error)

                self.back_propagate(error)

                for layer in self.layers:
                    layer.weights = layer.weights - gamma_t * layer.partials

            if epoch % 20 == 0 or epoch == 99:
                print("Epoch {} Sum Square Error: ".format(epoch), epoch_sum_square_error)

    def forward_propagate(self, inputs, print_steps=False):
        layer_inputs = inputs
        layer_activations = []
        for layer in self.layers:
            # activate layer using inputs
            layer_activations = layer.activate_layer(layer_inputs)

            # debug purposes
            if print_steps:
                print(layer_activations)

            # current layer activations = next layer inputs
            layer_inputs = layer_activations

        # return final layer activations
        return layer_activations

    def back_propagate(self, error):
        num_layers = len(self.layers)
        for i in reversed(range(num_layers)):

            # first iter this is the output layer
            current_layer = self.layers[i]

            layer_activations = current_layer.output
            layer_inputs = current_layer.inputs

            # output layer
            # if i == num_layers - 1:
            #     current_layer.partials = error * current_layer.inputs
            #     continue


            # credit: https://github.com/musikalkemist/DeepLearningForAudioWithPython ==============================
            # I got stuck on the back propagation algorithm.
            # this github users code helped me understand how to cache and use the
            # derivatives of the previous layer
            # code is heavily modified to fit my implementation


            delta = error * sigmoid_derivative(layer_activations)
            if i < num_layers - 1:
                delta = delta[0][0:-1].reshape((1, -1))

            # save derivative after applying matrix multiplication
            current_layer.partials = np.dot(layer_inputs.T, delta)

            # backpropogate the next error
            error = np.dot(delta, current_layer.weights.T)



    # Helper Functions ===========================================
    def build_net(self, net_structure, init_zero):
        num_layers = len(net_structure) - 1
        output_layer = False
        for i in range(num_layers):
            # mark the output layer
            if i == num_layers - 1:
                output_layer = True

            layer = Layer(net_structure[i], net_structure[i + 1], init_zero, output_layer)
            self.layers.append(layer)

    def square_error(self, error):
        return 0.5 * error ** 2

    def gamma_schedule(self, d, t):
        return self.gamma_0 / (1 + (self.gamma_0 / d) * t)


