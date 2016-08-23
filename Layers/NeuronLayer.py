import numpy as np


def get_probabilities_from_scores(scores):
    scores = np.minimum(200, scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class NeuronLayer:
    """Simple feed-forward layer"""

    activation_functions = dict(Sigmoid=lambda x: 1 / (1 + np.exp(-x)),
                                ReLU=lambda x: np.maximum(0, x),
                                TanH=np.tanh)

    activation_differentials = dict(Sigmoid=lambda x: x * (1 - x),
                                    ReLU=lambda x: 1 * (x > 0) + 0.01 * (x < 0),
                                    TanH=lambda x: 1 - x ** 2)

    def __init__(self, num_inputs, num_units, max_weight=0.1,
                 activation_function='Sigmoid'):
        self.__num_inputs = num_inputs
        self.__num_units = num_units
        self.__activation_function_type = activation_function
        self.__activation_function = self.activation_functions[activation_function]
        self.__inputs = None
        self.weights = np.random.randn(num_inputs, num_units) * max_weight
        self.bias = np.zeros((1, num_units))
        self.delta = None

    def get_inputs(self):
        return self.__inputs

    def set_inputs(self, inputs):
        self.__inputs = inputs

    def get_output(self):
        transfer_output = self.get_transfer_output()
        return self.__activation_function(transfer_output)

    def get_transfer_output(self):
        transfer_output = np.dot(self.__inputs, self.weights) + self.bias
        return transfer_output

    def calculate_cross_entropy_loss(self, targets):
        num_points = np.shape(targets)[0]
        output = self.get_transfer_output()
        probabilities = 0.001 + get_probabilities_from_scores(output)
        correct_log_probs = -np.log(probabilities[range(num_points),
                                                  targets])
        cross_entropy_loss = np.sum(correct_log_probs) / num_points
        return cross_entropy_loss

    def calculate_delta(self, other_layer_delta):
        self.delta = other_layer_delta * \
                     self.activation_differentials[self.__activation_function_type] \
                         (self.get_output())
        return self.delta
