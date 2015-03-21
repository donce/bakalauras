from sys import stdin

w = [0, 0, 0]

input = [1, 0, 0]

def error(value, expected):
    return (expected - value) ** 2

def f(value):
    return 1 if value >= 1 else 0

learning_rate = 0.3

while True:
    print '----------'
    print 'weights:', w
    print 'Enter input[1], input[2], expected_output:',
    try:
        input[1], input[2], expected_output = map(int, stdin.readline().split(' '))
    except (ValueError):
        print 'error in input!'
        continue
    print

    output = f(sum(input[i] * w[i] for i in xrange(3)))
    print 'output:', output
    e = error(output, expected_output)
    print

    if e:
        print 'Error, learning...'

        nw = w[:]
        for i, value in enumerate(w):
            delta_error = input[i] * (output - expected_output)
            delta = -learning_rate * delta_error
            nw[i] += delta
        w = nw
    else:
        print 'Correct'
