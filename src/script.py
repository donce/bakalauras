# -*- coding: utf-8 -*-
'''
Single layer perceptron using sigmoid activation function.
AND and OR in one network, using two output neurons.
'''
import math
import random
from datetime import datetime
from time import time
import pickle

from numpy import array, zeros, dot, vectorize, transpose, mean, std, copy as numpy_copy

import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D # for 3d graphs


matplotlib.rc('font', family='Arial')

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

def figname(name=None):
    now = datetime.now()
    return 'diagrams/{}{}-{}-{}_{}-{}-{}.pdf'.format(name + '_' if name else '', now.year, now.month, now.day, now.hour, now.minute, now.second)


class Network(object):
    """
    Attributes:
        biases [layer][j][0] float
        weights [layer][from][to] float

        activations [layer][j][0] float - outputing value after applying activation function
        inputs [layer][0][j] float - sum of inputs received
    """

    def __init__(self, learning_rate=0.5, momentum_coefficient=0.0, name=None, show_result=True):
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.name = name
        self.show_result = show_result

        self.inputs = []
        self.activations = []
        self.weights = []
        self.biases = []

        self.velocities = []

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
        network.name = self.name
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

        for layer in xrange(1, self.layers):
            w = self.weights[layer]
            a = self.activations[layer-1]
            self.inputs[layer] = dot(w, a) + self.biases[layer]
            self.activations[layer] = self.apply_activation(self.inputs[layer])

    def get_output(self):
        return self.activations[self.layers-1]

    # if desired_result is received, error is returned
    def run(self, inputs, desired_result=None):
        self.set_input_data(inputs)
        self.feed_forward()

        if desired_result is not None:
            assert self.layer_neurons(self.layers-1) == len(desired_result)
            total_error = 0
            result = self.get_output()
            for a in xrange(self.layer_neurons(self.layers-1)):
                output = result[a][0]
                desired_output = desired_result[a]

                error = (output - desired_output) ** 2
                total_error += error
            return total_error / len(desired_result)

    def run_encoding(self, data, encoding_layer):
        self.run(data)
        return numpy_copy(transpose(self.activations[encoding_layer])[0])

    def run_classification(self, data):
        self.run(data)
        return self.get_output().argmax()

    def teach_case(self, inputs, desired_result):
        output_layer = self.layers-1
        assert len(desired_result) == self.layer_neurons(output_layer)

        error = self.run(inputs, desired_result)

        desired_result = array([[result] for result in desired_result])
        result = self.get_output()

        deltas = self.layers * [None]
        deltas[self.layers-1] = 2 * (result - desired_result) * vectorized_sigmoid_derivative(result)
        for layer in reversed(range(1, self.layers-1)):
            deltas[layer] = dot(transpose(self.weights[layer+1]), deltas[layer+1]) * vectorized_sigmoid_derivative(self.activations[layer])

        for layer in xrange(1, self.layers):
            self.velocities[layer] = self.velocities[layer] * self.momentum_coefficient - self.learning_rate * deltas[layer]
            self.biases[layer] += self.velocities[layer]
            self.weights[layer] += dot(self.velocities[layer], transpose(self.activations[layer-1]))

        return error

    def teach(self, data, validation_data=None, allowed_error=0.0001, max_cycles=None, cycles=None):
        self.velocities = [zeros(a.shape) if a is not None else None for a in self.biases]
        best_network = self
        best_error = None

        success = False
        cycle = 0
        errors = []
        validation_errors = []
        while cycles is not None or not success:
            # self.learning_rate = 1.0 / (cycle / 10 + 2) # TODO: remove?
            success = True

            #learning
            cycle_error = 0
            for case in data:
                curr_error = self.teach_case(case[0], case[1])# TODO: diff in this line
                cycle_error += curr_error
                if curr_error > allowed_error:
                    success = False
            cycle += 1
            iteration_error = cycle_error / len(data)
            errors.append(iteration_error)

            #validation
            # TODO: extract common code with above
            if validation_data:
                cycle_error = 0
                for case in validation_data:
                    curr_error = self.run(case[0], case[1])# TODO: test
                    cycle_error += curr_error
                iteration_error = cycle_error / len(validation_data)
                validation_errors.append(iteration_error + 0.0)

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
        pyplot.xlabel(u'Mokymo iteracija', fontsize=AXIS_LABEL_FONT_SIZE)
        pyplot.ylabel(u'Klaida', fontsize=AXIS_LABEL_FONT_SIZE)
        pyplot.plot(errors, label=u'Mokymosi klaida')
        if validation_data:
            pyplot.plot(validation_errors, label=u'Validacijos klaida')
        pyplot.legend()
        pyplot.savefig(figname(self.name), format='pdf')
        if self.show_result:
            pyplot.show()
        else:
            pyplot.close()

        print 'success:', success
        # print 'cycles:', cycle
        # print 'finished'
        return best_error, best_network


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
    return [(d, d) for d in inputs]  # TODO: remove numpy_copy


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
        type = nr / 500
        input = map(int, line.strip().split())
        type_output = [1.0 if i == type else 0.0 for i in range(3)]
        data.append((input, type_output))
    f.close()
    return data

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


# data = read_input()[:1500]
# normalize_input(data)

# network = Network()
# network.generate((30, 3))
# network = Network()
# network.generate((30, 100, 100, 3)) #best for classification

# network = Network()
# network.generate((30, 2, 30))


data = read_input()[:1500]
normalize_gaussian(data)

partial_data_group_size = 500
partial_data = data[0:0+partial_data_group_size] + data[500:500+partial_data_group_size] + data[1000:1000+partial_data_group_size]
validation_data_group_size = 0
validation_data = data[0+partial_data_group_size:0+partial_data_group_size+validation_data_group_size] + \
    data[500+partial_data_group_size:500+partial_data_group_size+validation_data_group_size] + \
    data[1000+partial_data_group_size:1000+partial_data_group_size+validation_data_group_size]

compress_partial_data = [d[0] for d in partial_data]
compress_validation_data = [d[0] for d in validation_data]

#values
vx = []
vy = []

best_vx = []
best_vy = []

start_time = time()

ITERATIONS_COUNT = 5


for dimensions in range(4, 30+1):
    best_iterations_error = 1
    for iteration in range(ITERATIONS_COUNT):
        print '-----------------------------------------'
        print 'dimensions:', dimensions

        pickle_filename = 'compressed/{}.data'.format(dimensions)
        try:
            best_compression_error, compressed_data = pickle.loads(open(pickle_filename).read())
        except IOError:
            print 'compression...'
            network = Network(learning_rate=0.2, momentum_coefficient=0.4, name='compression', show_result=False)
            # network = Network(learning_rate=0.01, momentum_coefficient=0.4)
            network.generate((30, 30, dimensions, 30, 30))
            compression_layer = 2

            best_compression_error, network = network.teach(generate_encoding_data(compress_partial_data), max_cycles=500)# TODO: ~500 cycles?
            # _, network = network.teach(generate_encoding_data(compress_partial_data), max_cycles=1,
            #                            validation_data=generate_encoding_data(compress_validation_data))

            draw_points = True
            points = [[] for i in xrange(dimensions)]

            compressed_data = []

            for i, d in enumerate(compress_partial_data):
                r = network.run_encoding(d, compression_layer)
                assert len(r) == dimensions
                # if draw_points:
                #     for d in xrange(dimensions):
                #         points[d].append(r[d])
                compressed_data.append((r, partial_data[i][1]))

            normalize_gaussian(compressed_data)
            pickle.dump((best_compression_error, compressed_data), open(pickle_filename, 'w'))

        compressed_partial_data = compressed_data[0:50] + compressed_data[500:550] + compressed_data[1000:1050]

        #drawing compressed points
        # if draw_points and dimensions in [2, 3]:
        #     fig = pyplot.figure(figsize=(6, 6))
        #     if dimensions == 2:
        #         pyplot.xlabel('x', fontsize=AXIS_LABEL_FONT_SIZE)
        #         pyplot.ylabel('y', fontsize=AXIS_LABEL_FONT_SIZE)
        #         pyplot.plot(points[0][:500], points[1][:500], 'ro')
        #         pyplot.plot(points[0][500:1000], points[1][500:1000], 'go')
        #         pyplot.plot(points[0][1000:1500], points[1][1000:1500], 'bo')
        #     if dimensions == 3:
        #         ax = fig.add_subplot(111, projection='3d')
        #         ax.set_xlabel('x', fontsize=AXIS_LABEL_FONT_SIZE)
        #         ax.set_ylabel('y', fontsize=AXIS_LABEL_FONT_SIZE)
        #         ax.set_zlabel('z', fontsize=AXIS_LABEL_FONT_SIZE)
        #         ax.scatter(points[0][:500], points[1][:500], points[2][:500], c='r')
        #         ax.scatter(points[0][500:1000], points[1][500:1000], points[2][500:1000], c='g')
        #         ax.scatter(points[0][1000:1500], points[1][1000:1500], points[2][1000:1500], c='b')
        #     pyplot.savefig(figname('points'), format='pdf')
        #     pyplot.show()

        print 'classification...'
        cls_network = Network(learning_rate=0.2, momentum_coefficient=0.4, name='classification', show_result=False)
        cls_network.generate((dimensions, 30, 30, 3))
        best_current_error, cls_network = cls_network.teach(compressed_partial_data, max_cycles=400)
        # print 'best_current_error:', best_current_error
        # vx.append(dimensions)
        # vy.append(1.0 - best_current_error)

        curr_global_cls_error = 0.0
        for inputs, desired_result in compressed_data:
            curr_global_cls_error += cls_network.run(inputs, desired_result)
        curr_global_cls_error /= len(compressed_data)
        vx.append(dimensions)
        vy.append(1.0 - curr_global_cls_error)
        best_iterations_error = min(best_iterations_error, curr_global_cls_error)

        '''
        total = len(compressed_data)
        correct = 0
        for d, expected in compressed_data:
            result = cls_network.run_classification(d)
            expected_result = array(expected).argmax()
            if result == expected_result:
                correct += 1

        print 'result {}/{}'.format(correct, total)
        '''
    print 'best_iterations_error', best_iterations_error
    best_vx.append(dimensions)
    best_vy.append(1.0 - best_iterations_error)


elapsed_time = time() - start_time
print 'Elapsed time: {}s'.format(elapsed_time)

dimfig = pyplot.figure(figsize=(20, 12))
axis = pyplot.gca()
axis.set_xticks(vx)
pyplot.xlabel(u'Dimensijų skaičius', fontsize=AXIS_LABEL_FONT_SIZE)
pyplot.ylabel(u'1 - mokymosi klaida', fontsize=AXIS_LABEL_FONT_SIZE)
pyplot.plot(vx, vy, 'ro', label=u'Visi rezultatai')
pyplot.plot(best_vx, best_vy, label=u'Geriausi rezultatai')
pyplot.legend()
pyplot.savefig(figname('dimensions'), format='pdf')
pyplot.show()



# iris_data = read_iris()
# normalize_input(iris_data)

# random.shuffle(iris_data)

# network = Network()
# network.generate((4, 4, 4, 3), use_random=True)
# network.teach(iris_data, max_cycles=10)
