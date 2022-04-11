import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def test(inputs, outputs):
    correct = 0
    for i in range(len(inputs)):
        print('\nExample ', i)
        input = np.array(inputs[i], ndmin=2).T
        output_expected = training_outputs[0, i]

        weighted_sum_hidden = np.dot(wih, input) + bias
        output_hidden = sigmoid(weighted_sum_hidden)

        weighted_sum_output = np.dot(who, output_hidden)
        output_output = sigmoid(weighted_sum_output)
        output = 0.01
        if (output_output >= 0.5):
            output = 0.99
        if output != output_expected:
            print('\t Error in example ', i)
        else:
            correct += 1
        print('Output Neuronal Network', output_output)
        print('Expected Output ', output_expected)

    print('\n--- Accuracy--- : ', correct / len(inputs))


# input: x1=dog, x2=shorthair, x3=housebroken
training_inputs = np.array([[0, 1, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])
training_outputs = np.array([[0, 1, 1, 0]])

test_inputs = np.array([[0, 0, 1],
                        [1, 0, 0]])
test_outputs = np.array([[0, 0]])

# inputs within activiation function range
training_inputs = training_inputs * 0.98 + 0.01
test_inputs = test_inputs * 0.98 + 0.01

# outputs within activiation function range
training_outputs = training_outputs * 0.98 + 0.01
test_outputs = test_outputs * 0.98 + 0.01

bias = 0.2
L = 0.1
epochs = 200
np.random.seed(42)
wih = 2 * np.random.random((3, 3)) - 1
who = 2 * np.random.random((1, 3)) - 1
print(wih)
print(who)

for e in range(0, epochs):
    for i in range(len(training_inputs)):
        output_expected = np.array(training_outputs[:, i], ndmin=2).T
        input = np.array(training_inputs[i], ndmin=2).T

        # Forwardpropagation
        weighted_sum_hidden = np.dot(wih, input) + bias  ###
        output_hidden = sigmoid(weighted_sum_hidden)

        weighted_sum_output = np.dot(who, output_hidden)  ###
        output_output = sigmoid(weighted_sum_output)

        error = ((1 / 2) * (np.power((output_expected - output_output), 2)))

        # Backpropagation
        # Output->Hidden

        d_error_d_output_output = output_output - output_expected
        d_output_output_d_weightedsum_output = sigmoid_der(output_output)
        d_weightedsum_output_d_who = np.array(output_hidden, ndmin=2).T
        delta_who = np.dot(d_error_d_output_output * d_output_output_d_weightedsum_output, d_weightedsum_output_d_who)

        # Hidden->Input
        d_error_d_output_hidden = np.dot(np.array(who, ndmin=2).T,
                                         d_error_d_output_output * d_output_output_d_weightedsum_output)
        d_output_hidden_d_weightedsum_hidden = sigmoid_der(output_hidden)
        d_weightedsum_hidden_d_wih = np.array(input, ndmin=2).T
        delta_wih = np.dot(d_error_d_output_hidden * d_output_hidden_d_weightedsum_hidden, d_weightedsum_hidden_d_wih)

        # Update weights
        who = who - L * delta_who
        wih = wih - L * delta_wih

print("-------")
print('Weights_ih after epochs:\n', wih)
print('Weights_ho after epochs:\n', who)

print('\n Training')
test(training_inputs, training_outputs)

print('\n Testing')
test(test_inputs, test_outputs)

