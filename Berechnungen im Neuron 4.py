import numpy as np


def heaviside(x):
    if (x < 0):
        return 0
    else:
        return 1


# input: x1=dog, x2=shorthair, x3=housebroken
training_inputs = np.array([[0, 1, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])
training_outputs = np.array([[0, 1, 1, 0]])

bias = 0.2
L = 0.1
epochs = 20

np.random.seed(42)
weights = 2 * np.random.random((1, 3)) - 1
print(weights)

for e in range(0, epochs):
    for i in range(len(training_inputs)):
        print("-------")
        output_expected = np.array(training_outputs[:, i], ndmin=2).T
        input = np.array(training_inputs[i], ndmin=2).T
        # Forwardpropagation
        weighted_sum = np.dot(weights, input) + bias
        output = heaviside(weighted_sum)

        error = output_expected - output

        for n in range(0, len(weights[0, :])):
            print(' Before ', weights[:, n])
            weights[:, n] = weights[:, n] + error * L * training_inputs[i][n]
            print(' after ', weights[:, n])

print("-------")
print('Weights after epochs:\n', weights)

for i in range(len(training_inputs)):
    output_expected = np.array(training_outputs[:, i], ndmin=2).T
    input = np.array(training_inputs[i], ndmin=2).T

    weighted_sum = np.dot(weights, input) + bias
    output = heaviside(weighted_sum)
    if output != output_expected:
        print('Error in training example ', i)

