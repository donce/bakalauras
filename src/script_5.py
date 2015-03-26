'''
Single layer perceptron using sigmoid activation function.
AND and OR in one network, using two output neurons.

TODO:
XOR
'''
import math
import random

from numpy import array, dot, vectorize, transpose, empty


def sigmoid_function(value):
    return 1.0 / (1.0 + pow(math.e, -value))
vectorized_sigmoid_function = vectorize(sigmoid_function)


def sigmoid_derivative(value):
    return value * (1 - value)
vectorized_sigmoid_derivative = vectorize(sigmoid_derivative)


class Network(object):
    """
    Attributes:
        activations [layer][j][0] float
        biases [layer][j][0] float

        inputs [layer][0][j] int

        weights [layer][from][to] float
    """
    inputs = None
    activations = None

    weights = None
    biases = None


    learning_rate = None

    def __init__(self, layer_neurons, learning_rate=0.3):
        self.weights = []
        self.inputs = []
        self.activations = []
        self.biases = []
        for layer, neurons in enumerate(layer_neurons):
            self.biases.append(array([[0.0] for _ in xrange(neurons)]))
            self.activations.append(array([array([0.0]) for _ in xrange(neurons)]))
            self.inputs.append(array([[0.0 for _ in xrange(neurons)]]))

            if layer > 0:
                prev_neurons = layer_neurons[layer - 1]
                # curr_weights = array([[2.0 * random.random() - 1.0 for _ in xrange(prev_neurons)] for _ in xrange(neurons)])
                curr_weights = array([[0.0 for _ in xrange(prev_neurons)] for _ in xrange(neurons)])
            else:
                curr_weights = array([])
            self.weights.append(curr_weights)

        self.learning_rate = learning_rate

    def print_data(self):
        print 'activations:', self.activations
        print 'inputs:', self.inputs
        print 'biases:', self.biases

    def print_state(self):
        print 'w0 = {} w1 = {}'.format(self.weights[1][0][0], self.weights[1][0][1])
        print 'b = {}'.format(self.biases[1][0][0])

    @property
    def layers(self):
        return len(self.activations)

    def layer_neurons(self, layer):
        return len(self.activations[layer])

    def set_input_data(self, data):
        data = array([array([input]) for input in data])
        if data.shape != (self.layer_neurons(0), 1):
            raise ValueError
        self.activations[0] = data

    def _clean_inputs(self):
        for layer in xrange(1, self.layers):
            for i in xrange(self.layer_neurons(layer)):
                self.activations[layer][i] = 0

    def apply_activation(self, inputs):
        return vectorized_sigmoid_function(inputs)

    def feed_forward(self):
        self._clean_inputs()

        for layer in xrange(1, self.layers):
            w = self.weights[layer]
            a = self.activations[layer-1]
            self.inputs[layer] = dot(w, a) + self.biases[layer]
            self.activations[layer] = self.apply_activation(self.inputs[layer])
        print '------'

    def get_output(self):
        return self.activations[self.layers-1]

    def run(self, inputs, desired_result):
        desired_result = array([[result] for result in desired_result])
        output_layer = self.layers-1
        assert len(desired_result) == self.layer_neurons(output_layer)

        self.set_input_data(inputs)
        self.feed_forward()

        result = self.get_output()

        total_error = 0

        deltas = (self.layers) * [None]
        deltas[self.layers-1] = 2 * (result - desired_result) * vectorized_sigmoid_derivative(result)

        for layer in reversed(range(1, self.layers-1)):
            # deltas_last = deltas[self.layers-1]
            print 'LAYER:', layer
            print
            print 'NEXT'
            l = self.layers - 2
            print 'l =', l
            print 'w', self.weights[l+1]
            print 'wt', transpose(self.weights[l+1])
            deltas[layer] = dot(transpose(self.weights[l+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[l])
            print
        print 'DELTAS:', deltas

        for layer in range(1, self.layers-1):#TODO: +1, because we skip last layer - we do it in the code below
            print 'LEARNING', layer

        for i in xrange(self.layer_neurons(output_layer)):
            output = result[i][0]
            desired_output = desired_result[i][0]

            print 'output = {} (desired = {})'.format(output, desired_output)
            error = (output - desired_output) ** 2
            total_error += error
            print 'error = {}'.format(error)

            delta = deltas[self.layers-1][i][0]
            for j in xrange(self.layer_neurons(output_layer-1)):
                dw = delta * self.activations[output_layer-1][j][0]
                print 'dw{} = {}'.format(j, dw)
                print '1 0 0/1'
                print output_layer, i, j

                self.weights[output_layer][i][j] -= self.learning_rate * dw
            db = delta
            print 'db = {}'.format(db)
            self.biases[output_layer][i][0] -= self.learning_rate * db
        return total_error

    def answer(self, output):
        return 1 if output > 0.5 else 0

    def teach(self, data, allowed_error=0.05, max_cycles=None, cycles=None):
        success = False
        cycle = 0
        # while not success:
        while cycles != None or not success:
            print 'before'
            self.print_state()
            print '===================='
            success = True
            for case in data:
                print '-------'
                print case
                error = self.run(case[0], case[1])
                # if abs(error - allowed_error)
                if error > allowed_error:
                    success = False
            cycle += 1
            if not success and max_cycles is not None and cycle >= max_cycles:
                print 'stopping because max cycles limit exceeded'
                break
            if cycles >= cycle:
                print 'stopping because cycles number reached'
                break
        print 'success:', success
        print 'cycles:', cycle
        print 'finished'
        self.print_state()
        return success


OR = (
    ((0, 0), (0,)),
    ((0, 1), (1,)),
    ((1, 0), (1,)),
    ((1, 1), (1,)),
)


AND = (
    ((0, 0), (0,)),
    ((0, 1), (0,)),
    ((1, 0), (0,)),
    ((1, 1), (1,)),
)


AND_OR = (
    ((0, 0), (0, 0)),
    ((0, 1), (0, 1)),
    ((1, 0), (0, 1)),
    ((1, 1), (1, 1)),
)

XOR = (
    ((0, 0), (0,)),
    ((0, 1), (1,)),
    ((1, 0), (1,)),
    ((1, 1), (0,)),
)


network = Network((2, 1))
assert network.teach(AND)

network = Network((2, 1))
assert network.teach(OR)
network = Network((2, 2))
# network.weights[1][0][0] = network.weights[1][0][1] = network.weights[1][1][0] = network.weights[1][1][1] = network.biases[1][0] = network.biases[1][1] = 0
assert network.teach(AND_OR)
#result: 184


# network = Network((2, 2, 1))
# network.teach(XOR)


def read_iris():
    iris_data = []
    types = {}
    next_type_nr = 0
    f = open('iris.data')
    for line in f.readlines():
        data = line.rstrip().split(',')
        # print data
        input = map(float, data[:4])
        type = data[4]
        if type not in types:
            types[type] = [1.0 if i == next_type_nr else 0.0 for i in range(3)]
            next_type_nr += 1
        iris_data.append((input, types[type]))
    f.close()
    return iris_data

iris_data = read_iris()
print iris_data


def normalize_input(data):
    min_values = data[0][0][:]
    max_values = data[0][0][:]
    for current in data:
        input = current[0]
        for nr, value in enumerate(input):
            min_values[nr] = min(min_values[nr], value)
            max_values[nr] = max(max_values[nr], value)
    for current in data:
        input = current[0]
        for nr, value in enumerate(input):
            input[nr] = (value - min_values[nr]) / (max_values[nr] - min_values[nr])
normalize_input(iris_data)


network = Network((4, 3))
# network.teach(iris_data)
