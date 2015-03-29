'''

Single layer perceptron using sigmoid activation function.
AND and OR in one network, using two output neurons.


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
        biases [layer][j][0] float
        weights [layer][from][to] float

        activations [layer][j][0] float - outputing value after applying activation function
        inputs [layer][0][j] float - sum of inputs received
    """
    inputs = None
    activations = None

    weights = None
    biases = None


    learning_rate = None
    use_random = True

    def __init__(self, layer_neurons, learning_rate=0.5, use_random=True):
        self.weights = []
        self.inputs = []
        self.activations = []
        self.biases = []
        for layer, neurons in enumerate(layer_neurons):
            if use_random:
                self.biases.append(array([[2.0 * random.random() - 1.0] for _ in xrange(neurons)]))
            else:
                self.biases.append(array([[0.0] for _ in xrange(neurons)]))

            self.activations.append(array([array([0.0]) for _ in xrange(neurons)]))
            self.inputs.append(array([[0.0 for _ in xrange(neurons)]]))

            if layer > 0:
                prev_neurons = layer_neurons[layer - 1]
                if use_random:
                    curr_weights = array([[2.0 * random.random() - 1.0 for _ in xrange(prev_neurons)] for _ in xrange(neurons)])
                else:
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
        for layer in reversed(range(1, self.layers-1)):
            deltas[layer] = dot(transpose(self.weights[layer+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[layer])

        for layer in range(1, self.layers):
            self.biases[layer] -= self.learning_rate * deltas[layer]
            self.weights[layer] -= self.learning_rate * transpose(dot(self.activations[layer-1], deltas[layer]))

        for j in xrange(self.layer_neurons(output_layer)):
            output = result[j][0]
            desired_output = desired_result[j][0]

            error = (output - desired_output) ** 2
            total_error += error

        correct = False
        return correct, total_error

    def teach(self, data, allowed_error=0.01, max_cycles=None, cycles=None):
        success = False
        cycle = 0
        errors = []
        while cycles != None or not success:
            success = True
            cycle_error = 0
            for case in data:
                correct, curr_error = self.run(case[0], case[1])
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



#correct: 288
network = Network((2, 1), use_random=False)
network.teach(AND)

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
            input[nr] = float(value - min_values[nr]) / (max_values[nr] - min_values[nr])

'''
About data:
Each input has 50 integer values
Each group has 500 samples (0:500, 500:1500...)
'''


def read_input():
    data = []
    f = open('../data/CR1')
    for nr, line in enumerate(f.readlines()):
        type = nr / 50
        input = map(int, line.strip().split())
        type_output = [1.0 if i == type else 0.0 for i in range(3)]
        data.append((input, type_output))
    f.close()
    return data

# data = read_input()[:150]
# normalize_input(data)


# network = Network((30, 3))
# network = Network((30, 100, 100, 3)) #best
# network.teach(data, max_cycles=30)


def read_iris():
    iris_data = []
    types = {}
    next_type_nr = 0
    f = open('iris.data')
    for line in f.readlines():
        data = line.rstrip().split(',')
        input = map(float, data[:4])
        type = data[4]
        if type not in types:
            types[type] = [1.0 if i == next_type_nr else 0.0 for i in range(3)]
            next_type_nr += 1
        iris_data.append((input, types[type]))
    f.close()
    return iris_data



# iris_data = read_iris()
# normalize_input(iris_data)

# random.shuffle(iris_data)

# network = Network((4, 4, 4, 3), use_random=True)
# network.teach(iris_data, max_cycles=10)
