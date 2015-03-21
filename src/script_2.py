'''
Single layer perceptron using sigmoid activation function.
AND/OR networks.
'''
import math
import random


# def hadamard_product

class Network(object):
    inputs = None
    weights = None
    biases = None

    LEARNING_RATE = 1.0

    def __init__(self, layer_neurons):
        self.weights = []
        self.inputs = []
        self.biases = []
        for layer, neurons in enumerate(layer_neurons):
            self.biases.append([0 for i in xrange(neurons)])
            self.inputs.append([0 for i in xrange(neurons)])

            if layer + 1 < len(layer_neurons):
                next_neurons = layer_neurons[layer + 1]
                curr_weights = [[2 * random.random() - 1 for j in xrange(next_neurons)] for i in xrange(neurons)]
                self.weights.append(curr_weights)


    def print_state(self):
        print 'w1={} w2={}'.format(self.weights[0][0][0], self.weights[0][1][0])
        print 'b={}'.format(self.biases[1][0])

    @property
    def layers(self):
        return len(self.inputs)

    def layer_neurons(self, layer):
        return len(self.inputs[layer])

    def set_inputs(self, inputs):
        inputs = list(inputs)
        if len(inputs) != self.layer_neurons(0):
            raise ValueError
        self.inputs[0] = inputs

    def _clean_inputs(self):
        for layer in xrange(1, self.layers):
            for i in xrange(self.layer_neurons(layer)):
                self.inputs[layer][i] = self.biases[layer][i]#TODO: move to calculation

    def feed_forward(self):
        self._clean_inputs()
        for layer in xrange(self.layers-1):
            for neuron, input in enumerate(self.inputs[layer]):
                output = self.activation_function(input)
                # print '  output =', output
                for next_neuron in xrange(self.layer_neurons(layer+1)):
                    self.inputs[layer+1][next_neuron] += output * self.weights[layer][neuron][next_neuron]

    def get_output(self):
        inputs = self.inputs[self.layers-1]
        return [self.activation_function(input) for input in inputs]

    def activation_function(self, value):
        return 1.0 / (1.0 + pow(math.e, -value))

    def sigmoid_derivative(self, value):
        return value * (1 - value)

    def run(self, inputs, desired_output):
        self.set_inputs(inputs)
        self.feed_forward()

        output = self.get_output()[0]
        print 'output:', output
        print 'desired_output', desired_output
        error = (output - desired_output) ** 2
        print 'error:', error

        delta = 2 * (output - desired_output) * output * (1 - output)
        print 'delta', delta
        for i in xrange(self.layer_neurons(0)):
            dw = delta * self.inputs[0][i]
            print 'dw{}={}'.format(i, dw)
            self.weights[0][i][0] -= self.LEARNING_RATE * dw
        db = delta
        print 'db={}'.format(db)
        self.biases[1][0] -= self.LEARNING_RATE * db
        return self.answer(output)

    def answer(self, output):
        return 1 if output > 0.5 else 0

    def teach(self, data, allowed_error=0, max_cycles=None, cycles=None):
        # print data
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
                # self.set_inputs(case[0])
                answer = self.run(case[0], case[1])
                # if abs(error - allowed_error)
                if answer != case[1]:#error > allowed_error:
                    success = False
                # print self.weights[0][0][0], self.weights[0][1][0]
            cycle += 1
            if not success and max_cycles is not None and cycle >= max_cycles:
                print 'stopping because max cycles limit exceeded'
                break
            if cycles >= cycle:
                print 'stopping because cycles number reached'
                break
            # if cycle >= max_cycles:
            #     break
        print 'success:', success
        print 'cycles:', cycle
        print 'finished'
        self.print_state()
        return success


AND = (
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
)

OR = (
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 1),
)

network = Network((2, 1))
assert network.teach(OR)

network = Network((2, 1))
assert network.teach(AND)
