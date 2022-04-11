import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# input: x1=dog, x2=shorthair, x3=housebroken
training_inputs = np.array([[0, 1, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])
training_outputs = np.array([[0, 1, 1, 0]])

# inputs within activiation function range
training_inputs = training_inputs * 0.98 + 0.01

# outputs within activiation function range
training_outputs = training_outputs * 0.98 + 0.01

bias = 0.2
L = 0.1
epoches = 2000
np.random.seed(42)
weights = 2 * np.random.random((1, 3)) - 1
print(weights)

for e in range(0, epoches):
    for i in range(len(training_inputs)):
        output_expected = np.array(training_outputs[:, i], ndmin=2).T
        input = np.array(training_inputs[i], ndmin=2).T

        # Forwardpropagation
        weighted_sum = np.dot(weights, input) + bias
        output = sigmoid(weighted_sum)

        error = ((1 / 2) * (np.power((output_expected - output), 2)))

        # Backpropagation
        # d entspricht dem delta der mathematischen Form
        d_error_d_output_output = output - output_expected      # Ableitung des Fehlers zur Ableitung des Outputs
                                                                # d.h.: Berechnete Fehler - Output
        d_output_output_d_weightedsum = sigmoid_der(output)     # Ableitung des Outputs nach der gew. Summe
                                                                # Ableitung der Aktivierungsfunktion vom output
        d_weightedsum_d_weights = np.array(input, ndmin=2).T    # Ableitung der gew. Summe nach den Gewichten
                                                                # Input ist [3, 1]-Matrix, wir brauchen aber [1,3], deshalb T
                                                                # die Dimensionen müssen passen, deshalb den Input transponieren
                                                                # Array mit 2 Dimensionen (ndmin=2)
        # Gradient wird zusammengesetzt!
        delta_weights = np.dot(d_error_d_output_output * d_output_output_d_weightedsum, d_weightedsum_d_weights)
        # multipliziert den d_error_d_output_output * d_output_output_d_weightedsum
        # bildet dan das Skalarprodukt mit dem Ergebnis der Multiplikation mit d_weightedsum_d_weights

        # Update weights        Gewichte anpassen!
        weights = weights - L * delta_weights

print("-------")
print('Weights after epoches:\n', weights)

correct = 0
for i in range(len(training_inputs)):
    output_expected = np.array(training_outputs[:, i], ndmin=2).T
    input = np.array(training_inputs[i], ndmin=2).T

    weighted_sum = np.dot(weights, input) + bias
    output_output = sigmoid(weighted_sum)

    output = 0.01
    if (output_output >= 0.5):
        output = 0.99
    if output != output_expected:
        print('\t Error in example ', i)
    else:
        correct += 1
    print('Output Neuronal Network', output_output)
    print('Expected Output ', output_expected)

print('\n--- Accuracy--- : ', correct / len(training_inputs))

