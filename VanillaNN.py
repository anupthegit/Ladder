import numpy as np

import Layers
from Layers import NeuronLayer


def evaluate_network(outputs, targets):
    assert len(outputs) == len(targets), \
        "Different number of points for output and target!"
    accuracy = np.mean(outputs == targets)
    return accuracy


def class_labels_to_one_hot(labels, num_classes):
    num_points = np.shape(labels)[0]
    one_hots = np.zeros((num_points, num_classes))
    for i in range(num_points):
        one_hots[i][labels[i]] = 1
    return one_hots


class VanillaNN:
    """Vanilla neural network"""

    def __init__(self, num_inputs, num_outputs,
                 num_hidden_layers, hidden_layer_sizes,
                 layer_type='Sigmoid', reg_param=0):
        """
        Initialize feed-forward neural net

        :arg num_inputs: number of inputs in input layer
        :arg num_outputs: number of outputs in output layer
        :arg num_hidden_layers: number of layers as integer
        :arg hidden_layer_sizes: layer sizes as array
        """
        self.__num_outputs = num_outputs
        self.__num_inputs = num_inputs
        self.__num_hidden_layers = num_hidden_layers
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.layers = None
        self.__input_layer = np.matrix(np.empty((self.__num_inputs + 1, 1)))
        self.__output_layer = np.matrix(np.empty((self.__num_outputs, 1)))
        self.__layer_outputs = []
        self.__layer_type = layer_type
        self.reg_param = reg_param
        self.init_layers()

    def init_layers(self):
        assert self.__num_hidden_layers == \
               np.shape(self.__hidden_layer_sizes)[0], \
            "Layer sizes array size should be number of layers!"
        # Layer sizes including input and output layers
        layer_sizes = self.__hidden_layer_sizes + [self.__num_outputs]
        layer_sizes.insert(0, self.__num_inputs)
        self.layers = [NeuronLayer(layer_sizes[i], layer_sizes[i + 1],
                                   activation_function=self.__layer_type)
                       for i in range(self.__num_hidden_layers + 1)]

    def forward_propagate(self, inputs):
        layer_input = inputs
        for layer in self.layers:
            layer.set_inputs(layer_input)
            layer_input = layer.get_output()

    def calculate_cost(self, targets):
        return self.layers[-1].calculate_cross_entropy_loss(targets)

    def calculate_reg_loss_gradients(self, gradients):
        reg_loss = 0.5 * self.reg_param * \
                   sum([np.sum(l.weights ** 2) for l in self.layers])
        reg_gradients = [gradient + self.reg_param * layer.weights
                         for gradient, layer in zip(gradients, self.layers)]
        return reg_loss, reg_gradients

    def back_propagate(self, targets):
        weight_gradients, bias_gradients = [], []
        other_layer_delta = self.output_layer_delta(targets)
        forward_weights = np.eye(self.__num_outputs)
        for layer in reversed(self.layers):
            other_layer_delta = np.dot(other_layer_delta, forward_weights.T)
            other_layer_delta = layer.calculate_delta(other_layer_delta)
            layer_weight_gradient = np.dot(np.transpose(layer.get_inputs()),
                                           other_layer_delta)
            weight_gradients.insert(0, layer_weight_gradient)
            layer_bias_gradient = np.sum(other_layer_delta, axis=0,
                                         keepdims=True)
            bias_gradients.insert(0, layer_bias_gradient)
            forward_weights = layer.weights
        return weight_gradients, bias_gradients

    def output_layer_delta(self, targets):
        output_layer_scores = self.layers[-1].get_transfer_output()
        output_layer_probabilities = Layers.get_probabilities_from_scores(
            output_layer_scores
        )
        output_layer_probabilities[range(np.shape(targets)[0]), targets] -= 1
        other_layer_delta = output_layer_probabilities / np.shape(targets)[0]
        return other_layer_delta

    def train_network(self, inputs, targets, epochs=10000, alpha=1,
                      batch_size=100):
        num_points = np.shape(inputs)[0]
        assert num_points == np.shape(targets)[0], \
            "Different number of input and target points!"
        for epoch in range(epochs):
            batch_indices = np.random.choice(num_points,
                                             batch_size, replace=False)
            self.forward_propagate(inputs[batch_indices])
            cost = self.calculate_cost(targets[batch_indices])
            weight_gradients, bias_gradients = self.back_propagate(targets
                                                                   [batch_indices])
            reg_loss, reg_gradients = self.calculate_reg_loss_gradients(weight_gradients)
            cost += reg_loss
            weight_gradients = reg_gradients
            if epoch % 1000 == 0:
                print "Cost at iteration {0} is {1}".format(epoch, cost)
            self.update_weights(weight_gradients, bias_gradients, alpha)

    def test_network(self, inputs):
        self.forward_propagate(inputs)
        output_layer_activations = self.layers[-1].get_output()
        return [np.argmax(i) for i in output_layer_activations]

    def update_weights(self, weight_gradients,
                       bias_gradients, alpha):
        assert len(self.layers) == len(weight_gradients) and \
               len(self.layers) == len(bias_gradients), \
            "Different number of gradient and weight/bias layers!"
        n = len(self.layers)
        for i in range(n):
            layer = self.layers[i]
            layer.weights += -alpha * weight_gradients[i]
            layer.bias += -alpha * bias_gradients[i]
