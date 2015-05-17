'''
Single layer perceptron using sigmoid activation function.
AND and OR in one network, using two output neurons.
'''
import math
import random

from numpy import array, zeros, dot, vectorize, transpose, mean, std, copy as numpy_copy

import matplotlib.pyplot as pyplot


AXIS_LABEL_FONT_SIZE = 20

def sigmoid_function(value):
    return 1.0 / (1.0 + pow(math.e, -value))
vectorized_sigmoid_function = vectorize(sigmoid_function)


def sigmoid_derivative(value):
    return value * (1.0 - value)
vectorized_sigmoid_derivative = vectorize(sigmoid_derivative)


def answer(output):
    return 1 if output > 0.5 else 0
vectorized_answer = vectorize(answer)


class Network(object):
    """
    Attributes:
        biases [layer][j][0] float
        weights [layer][from][to] float

        activations [layer][j][0] float - outputing value after applying activation function
        inputs [layer][0][j] float - sum of inputs received
    """
    inputs = []
    activations = []

    weights = []
    biases = []

    learning_rate = None
    momentum_coefficient = None

    def __init__(self, learning_rate=0.5, momentum_coefficient=0.0):
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient

    def generate(self, layer_neurons, use_random=True):
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

    def clone(self):
        network = Network()
        network.inputs = [numpy_copy(a) for a in self.inputs]
        network.activations = [numpy_copy(a) for a in self.activations]
        network.weights = [numpy_copy(a) for a in self.weights]
        network.biases = [numpy_copy(a) for a in self.biases]
        network.learning_rate = self.learning_rate
        return network

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

        # self.activations[0] = self.apply_activation(self.inputs[0])#foo

        for layer in xrange(1, self.layers):
            w = self.weights[layer]
            a = self.activations[layer-1]
            self.inputs[layer] = dot(w, a) + self.biases[layer]
            self.activations[layer] = self.apply_activation(self.inputs[layer])

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

        deltas = self.layers * [None]
        deltas[self.layers-1] = 2 * (result - desired_result) * vectorized_sigmoid_derivative(result)
        for layer in reversed(range(1, self.layers-1)):
            deltas[layer] = dot(transpose(self.weights[layer+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[layer])

        velocities = [zeros(a.shape) if a is not None else None for a in deltas]

        for layer in xrange(1, self.layers):
            velocities[layer] = velocities[layer] * self.momentum_coefficient - self.learning_rate * deltas[layer]
            self.biases[layer] += velocities[layer]
            self.weights[layer] += dot(velocities[layer], transpose(self.activations[layer-1]))

        for j in xrange(self.layer_neurons(output_layer)):
            output = result[j][0]
            desired_output = desired_result[j][0]

            error = (output - desired_output) ** 2
            total_error += error

        correct = False
        return correct, total_error / len(desired_result)

    def run_encoding(self, data, encoding_layer):
        self.run(data, data)
        return transpose(self.activations[encoding_layer])[0]

    def teach(self, data, allowed_error=0.01, max_cycles=None, cycles=None):
        best_network = self
        best_error = None

        success = False
        cycle = 0
        errors = []
        while cycles is not None or not success:
            success = True
            cycle_error = 0
            for case in data:
                correct, curr_error = self.run(case[0], case[1])
                cycle_error += curr_error
                if curr_error > allowed_error:
                    success = False
            cycle += 1
            iteration_error = cycle_error / len(data)
            errors.append(iteration_error)
            if (iteration_error < best_error) or not best_error:
                best_error = iteration_error
                best_network = self.clone()
            if not success and max_cycles is not None and cycle >= max_cycles:
                print 'stopping because max cycles limit exceeded'
                break
            if cycles >= cycle:
                print 'stopping because cycles number reached'
                break
        print len(errors)
        pyplot.xlabel('Mokymo iteracija', fontsize=AXIS_LABEL_FONT_SIZE)
        pyplot.ylabel('Klaida', fontsize=AXIS_LABEL_FONT_SIZE)
        pyplot.plot(errors)
        pyplot.show()

        print 'success:', success
        print 'cycles:', cycle
        print 'finished'
        self.print_state()
        return success, best_network


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
# network = Network()
# network.generate((2, 1), use_random=False)
# network.teach(AND)


def normalize_input_linear(data):
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


def normalize_gaussian(data):
    inputs = array([d[0] for d in data])
    means = mean(inputs, 0)
    stds = std(inputs, 0)
    for line in data:
        for j in range(len(line[0])):
            print j
            if stds[j] != 0:
                line[0][j] = float(line[0][j] - means[j]) / stds[j]
            else:
                line[0][j] = 0

'''
foo = array((
    (1, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
    (100, 2, 3),
), float)
'''

# normalize_input_1(foo)

def generate_encoding_data(inputs):
    return [(d, d) for d in inputs]


# for a in generate_encoding_data(DATA):
#     print a
# network = Network()
# network.generate((3, 2, 3), use_random=False)
# network.teach(generate_encoding_data(DATA), allowed_error=0.2)
# network.run_encoding(DATA[0])
# network.run_encoding(DATA[0])

'''
xs = []
ys = []

for d in DATA:
    x, y = network.run_encoding(d)
    xs.append(x)
    ys.append(y)
    print x, y
    # pyplot.
pyplot.plot(xs, ys, 'ro')
pyplot.show()
'''



'''
About data:
Each input has 30 integer values
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

# data = read_input()[:1500]
# normalize_input(data)

# network = Network()
# network.generate((30, 3))
# network = Network()
# network.generate((30, 100, 100, 3)) #best for classification

# network = Network()
# network.generate((30, 2, 30))
network = Network(learning_rate=0.2, momentum_coefficient=0.1)
network.generate((30, 30, 2, 30, 30))

data = read_input()[:1500]
normalize_input_linear(data)
data = [d[0] for d in data]

partial_data = data[0:50] + data[500:550] + data[1000:1050]

_, network = network.teach(generate_encoding_data(partial_data), max_cycles=1000)


xs = []
ys = []

for d in partial_data:
    x, y = network.run_encoding(d, 2)
    xs.append(x)
    ys.append(y)

pyplot.xlabel('x', fontsize=AXIS_LABEL_FONT_SIZE)
pyplot.ylabel('y', fontsize=AXIS_LABEL_FONT_SIZE)
pyplot.plot(xs[:50], ys[:50], 'ro')
pyplot.plot(xs[50:100], ys[50:100], 'go')
pyplot.plot(xs[100:150], ys[100:150], 'bo')
pyplot.show()


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

# network = Network()
# network.generate((4, 4, 4, 3), use_random=True)
# network.teach(iris_data, max_cycles=10)
