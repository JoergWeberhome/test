import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuronalNetwork:
    def __init__(self, no_input_nodes, no_hidden_nodes, no_output_nodes, L, bias):
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.L = L
        self.bias = bias
        self.wih = 2 * np.random.random((no_hidden_nodes, no_input_nodes)) - 1
        self.who = 2 * np.random.random((no_output_nodes, no_hidden_nodes)) - 1

    def train(self, training_images, training_labels):
        # trains for each training example the neuronal network by adapting the
        # weights based on gradient descent
        for i in range(len(training_images)):
            input = np.array(training_images[i], ndmin=2).T             # 785 Spalten 100 Zeilen
            output_expected = np.array(training_labels[i], ndmin=2).T   # 100 Spalten 1 Zeilen

            # Forwardpropagation
            weighted_sum_hidden = np.dot(self.wih, input)
            output_hidden = sigmoid(weighted_sum_hidden)

            weighted_sum_output = np.dot(self.who, output_hidden)
            output_output = sigmoid(weighted_sum_output)

                # Fehlerberechnung  1/2 * (erwarteter Wert - tatsaechl.Wert)^2
            error = ((1 / 2) * (np.power((output_expected - output_output), 2 )))

            ## Backpropagation
            # Output->Hidden
            d_error_d_output_output = output_output - output_expected
            d_output_output_d_weightedsum_output = sigmoid_der(output_output)
            d_weightedsum_output_d_who = np.array(output_hidden, ndmin=2).T
            delta_who = np.dot(d_error_d_output_output * d_output_output_d_weightedsum_output,
                               d_weightedsum_output_d_who)

            # Hidden->Input
            d_error_d_output_hidden = np.dot(np.array(self.who, ndmin=2).T,
                                             d_error_d_output_output * d_output_output_d_weightedsum_output)
            d_output_hidden_d_weightedsum_hidden = sigmoid_der(output_hidden)
            d_weightedsum_hidden_d_wih = np.array(input, ndmin=2).T
            delta_wih = np.dot(d_error_d_output_hidden * d_output_hidden_d_weightedsum_hidden,
                               d_weightedsum_hidden_d_wih)

            # Update weights
            self.who = self.who - self.L * delta_who
            self.wih = self.wih - self.L * delta_wih

    def evaluate(self, data, labels):
        # generates the confusion matrix
        confusion_matrix = np.zeros((self.no_output_nodes, self.no_output_nodes), int)
        for i in range(len(data)):
            test_result = self.test(data[i])
            test_result_max = test_result.argmax()
            target_label = labels[i].argmax()
            confusion_matrix[test_result_max, target_label] += 1

        print('Confusion matrix:\n', confusion_matrix)
        return confusion_matrix

    def test(self, input):
        # forward propagates the input and returns the computed output
        input = np.array(input, ndmin=2).T

        weighted_sum_hidden = np.dot(self.wih, input) + self.bias
        output_hidden = sigmoid(weighted_sum_hidden)

        weighted_sum_output = np.dot(self.who, output_hidden)
        output_output = sigmoid(weighted_sum_output)

        return output_output

    def measurements(self, confusion_matrix):
        # prints common neural network measures, i.e. accuracy, recall, precision
        no_correct, no_incorrect, accuracy = 0, 0, 0
        for i in range(self.no_output_nodes):
            no_correct += confusion_matrix[i][i]
            for j in range(self.no_output_nodes):
                if (i != j and confusion_matrix[i][j] != 0):
                    no_incorrect += confusion_matrix[i][j]

        for i in range(self.no_output_nodes):
            print("digit: ", i, "precision: ", self.precision(i, confusion_matrix), "recall: ",
                  self.recall(i, confusion_matrix))

        accuracy = no_correct / (no_correct + no_incorrect)

        return no_correct, no_incorrect, accuracy

    def recall(self, label, confusion_matrix):
        column = confusion_matrix[:, label]
        sum = column.sum()
        if (sum != 0):
            return confusion_matrix[label, label] / sum
        else:
            return 0

    def precision(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        sum = row.sum()
        if (sum != 0):
            return confusion_matrix[label, label] / sum
        else:
            return 0

#### data preparation ##############

#TODO: Determine number of input and output neurons
no_outputs = 10     # i.e. 0, 1, 2, 3, ..., 9
no_inputs = 28*28   # width times length = no of pixels for input

#read input
'''
all = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None)).values
inputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[*range(2, 32)])).values
outputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[1])).values
'''
training_inputs = np.loadtxt("MNIST_CSV/mnist_test_100.csv",
                             delimiter=',')
print('-----------------')
print('Training_inputs:',training_inputs)
print('Training_inputs:',training_inputs[:,0], 'Länge:',len(training_inputs[:,0]))
print('Training_inputs: Länge:',len(training_inputs[0,:]))  # Länge = 785: 1 = Label Rest 28*28 = 784
test_inputs = np.loadtxt("MNIST_CSV/mnist_train_1000.csv",
                         delimiter=",")

# Extracts the labels from the input files
training_outputs_tmp = np.asfarray(training_inputs[:,0]).astype(int)    # auf integer casten
test_outputs_tmp = np.asfarray(test_inputs[:,0]).astype(int)

# Transforms the label into an representation where the value of
# the corresponding output neuron has the value 0.99 (activated), while all other
# neurons have a value of 0.01 (not activated)
training_outputs = np.zeros((len(training_outputs_tmp), no_outputs)) + 0.01
for row in range(training_outputs_tmp.size):
    value = training_outputs_tmp[row]
    training_outputs[row][value] = 0.99

test_outputs = np.zeros((len(test_outputs_tmp), no_outputs)) + 0.01
for row in range(test_outputs_tmp.size):
    value = test_outputs_tmp[row]
    test_outputs[row][value] = 0.99

# Input values should be between 0.01 and 0.99
training_inputs = np.asfarray(training_inputs[:,1:])/255 * 0.98 + 0.01
test_inputs = np.asfarray(test_inputs[:,1:]) /255 * 0.98 + 0.01

# Shows the images of the digits
debug = False
if debug:
    for i in range(10):
        img = training_inputs[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.title(training_outputs_tmp[i])
        plt.show()

epochs = 100
hidden_nodes = 100
L = 0.1
bias = 0.1

print("-------------------------")
# Init new neural network
neuronal_network = NeuronalNetwork(no_inputs, hidden_nodes, no_outputs, L, bias)
for epoch in range(epochs):
    neuronal_network.train(training_inputs, training_outputs)

print("TRAINING SET")
confusion_matrix_training = neuronal_network.evaluate(training_inputs, training_outputs)
correct, incorrect, accuracy = neuronal_network.measurements(confusion_matrix_training)
print("Accuracy: ", accuracy)
print("Number of correct instances: ", correct)
print("Number of incorrect instances: ", incorrect)

print("TESTING SET")
confusion_matrix = neuronal_network.evaluate(test_inputs, test_outputs)
correct, incorrect, accuracy = neuronal_network.measurements(confusion_matrix)
print("Accuracy: ", accuracy)
print("Number of correct instances: ", correct)
print("Number of incorrect instances: ", incorrect)