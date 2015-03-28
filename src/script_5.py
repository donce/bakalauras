'''
Single layer perceptron using sigmoid activation function.
AND and OR in one network, using two output neurons.

TODO:
XOR
'''
import math
import random

from numpy import array, dot, vectorize, transpose, empty

import matplotlib.pyplot as pyplot


def sigmoid_function(value):
    return 1.0 / (1.0 + pow(math.e, -value))
vectorized_sigmoid_function = vectorize(sigmoid_function)


def sigmoid_derivative(value):
    return value * (1 - value)
vectorized_sigmoid_derivative = vectorize(sigmoid_derivative)


def answer(output):
    return 1 if output > 0.5 else 0
vectorized_answer = vectorize(answer)

# from winsound import Beep
# Beep(1000, 2000)


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

    def __init__(self, layer_neurons, learning_rate=0.001):
        self.weights = []
        self.inputs = []
        self.activations = []
        self.biases = []
        for layer, neurons in enumerate(layer_neurons):
            # self.biases.append(array([[2.0 * random.random() - 1.0] for _ in xrange(neurons)]))
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
        print
        print '======================='
        print 'State'
        print '======================='
        for layer in range(1, self.layers):
            print 'layer {}:'.format(layer)
            for curr in range(self.layer_neurons(layer)):
                print '   b = {}'.format(self.biases[layer][curr][0])
                for prev in range(self.layer_neurons(layer-1)):
                    print '   w{}, {} = {}'.format(curr, prev, self.weights[layer][curr][prev])
        print '======================='
        print

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
        # print '------'

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
        # print deltas
        for layer in reversed(range(1, self.layers-1)):
            # print 'Layer', layer

            # l = self.layers - 2
            # deltas[layer] = dot(transpose(self.weights[l+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[l])

            deltas[layer] = dot(transpose(self.weights[layer+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[layer])


            # print deltas
            # print
        # print 'DELTAS:', deltas

        for layer in range(1, self.layers-1):#TODO: +1, because we skip last layer - we do it in the code below
            # print 'LEARNING', layer
            delta = deltas[layer]
            # print delta
            # print self.biases[layer]
            self.biases[layer] -= self.learning_rate * delta
            # print self.biases[layer]
            for i in xrange(self.layer_neurons(layer)):
                for j in xrange(self.layer_neurons(layer-1)):
                    dw = deltas[layer][j][0]
                    # print 'dw{}, {} = {}'.format(i, j, dw)
                    self.weights[layer][i][j] -= self.learning_rate * dw

        for i in xrange(self.layer_neurons(output_layer)):
            output = result[i][0]
            desired_output = desired_result[i][0]

            # print 'output = {} (desired = {})'.format(output, desired_output)
            error = (output - desired_output) ** 2
            total_error += error
            # print 'error = {}'.format(error)

            delta = deltas[self.layers-1][i][0]
            for j in xrange(self.layer_neurons(output_layer-1)):
                dw = delta * self.activations[output_layer-1][j][0]
                # print 'dw{} = {}'.format(j, dw)
                self.weights[output_layer][i][j] -= self.learning_rate * dw
            # print 'db = {}'.format(delta)
            self.biases[output_layer][i][0] -= self.learning_rate * delta
        # correct = self.answer(self.)
        # print vectorized_answer(result)
        # print desired_result
        correct = False
        return correct, total_error

    def teach(self, data, allowed_error=0.01, max_cycles=None, cycles=None):
        success = False
        cycle = 0
        errors = []
        # while not success:
        while cycles != None or not success:
            # print 'before'
            # self.print_state()
            # print '===================='
            success = True
            cycle_error = 0
            for case in data:
                # print '-------'
                # print case
                correct, curr_error = self.run(case[0], case[1])
                # if abs(error - allowed_error)
                cycle_error += curr_error
                if curr_error > allowed_error:
                    success = False
            cycle += 1
            errors.append(cycle_error)
            if not success and max_cycles is not None and cycle >= max_cycles:
                print 'stopping because max cycles limit exceeded'
                break
            if cycles >= cycle:
                print 'stopping because cycles number reached'
                break
        print len(errors)
        pyplot.plot(errors)
        pyplot.show()

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


# network = Network((2, 2, 1))
# network = Network((2, 3, 1))
# network = Network((2, 2, 1))
# network.teach(XOR, max_cycles=1000)
# network.print_state()
# network.run(XOR[0][0], XOR[0][1])
# network.print_state()
# network.run(XOR[0][0], XOR[0][1])
# network.print_state()


# network = Network((2, 1))
# network.teach(AND)

# network = Network((2, 1))
# assert network.teach(OR)
# network = Network((2, 2))
# assert network.teach(AND_OR)
#result: 184


# network = Network((2, 2, 1))
# network.run(*XOR[0])
# network.run(*XOR[0])
# network.run(*XOR[0])
# network.run(*XOR[0])
# network.teach(XOR, max_cycles=5000)


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


iris_data = read_iris()
normalize_input(iris_data)
from random import shuffle
# shuffle(iris_data)

# network = Network((4, 3))
network = Network((4, 4, 4, 3))

# print network.run(iris_data[0][0], iris_data[0][1])
# print network.run(iris_data[0][0], iris_data[0][1])
print network.run(iris_data[0][0], iris_data[0][1])
# print network.run(iris_data[0][0], iris_data[0][1])
# network.teach(iris_data, max_cycles=100)
