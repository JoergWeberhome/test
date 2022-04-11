import numpy as np
import pandas as pd
import math
from scipy.special import expit


def sigmoid(x):
    #Otherwise  RuntimeWarning: overflow encountered in exp
    return expit(x)

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:

    def __init__(self, no_input_nodes, no_hidden_nodes, no_output_nodes,
                 L, bias):
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.L = L
        self.bias = bias
        self.wih = 2 * np.random.random((no_hidden_nodes, no_input_nodes)) - 1
        self.who = 2 * np.random.random((no_output_nodes, no_hidden_nodes)) - 1



    def train(self, training_images, training_labels):
        for i in range(len(training_images)):
            # inputs
            input = np.array(training_images[i], ndmin=2).T
            output_expected = np.array(training_labels[i], ndmin=2).T

            # Forwardpropagation
            weighted_sum_hidden = np.dot(self.wih, input) + self.bias
            output_hidden = sigmoid(weighted_sum_hidden)

            weighted_sum_output = np.dot(self.who, output_hidden)
            output_output = sigmoid(weighted_sum_output)

            error = ((1 / 2) * (np.power((output_expected - output_output), 2)))

            ##Bachpropagation
            #Output->Hidden
            d_error_d_output_output = output_output - output_expected
            d_output_output_d_weightedsum_output=sigmoid_der(output_output)
            d_weightedsum_output_d_who=np.array(output_hidden, ndmin=2).T
            delta_who= np.dot(d_error_d_output_output * d_output_output_d_weightedsum_output, d_weightedsum_output_d_who)

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
        confusion_matrix = np.zeros((self.no_output_nodes, self.no_output_nodes), int)
        for i in range(len(data)):
            test_result = self.test(data[i])
            test_result_max = test_result.argmax()
            target_label = labels[i].argmax()
            confusion_matrix[test_result_max, target_label] += 1
        print('Confusion matrix:\n', confusion_matrix)
        return confusion_matrix


    def test(self, input):
        input = np.array(input, ndmin=2).T

        weighted_sum_hidden = np.dot(self.wih, input) + self.bias
        output_hidden = sigmoid(weighted_sum_hidden)

        weighted_sum_output = np.dot(self.who, output_hidden)
        output_output = sigmoid(weighted_sum_output)

        return output_output

    def measurements(self, confusion_matrix):
        no_correct, no_incorrect, accuracy = 0,0,0
        for i in range(self.no_output_nodes):
            no_correct += confusion_matrix[i][i]
            for j in range(self.no_output_nodes):
                if(i!=j and confusion_matrix[i][j]!=0):
                    no_incorrect+=confusion_matrix[i][j]

        for i in range(self.no_output_nodes):
            print("precision: ", self.precision(i, confusion_matrix), "recall: ", self.recall(i, confusion_matrix))

        accuracy = no_correct / (no_correct + no_incorrect)

        return no_correct, no_incorrect, accuracy

    def recall(self, label, confusion_matrix):
        column = confusion_matrix[:, label]
        sum=column.sum()
        if(sum!=0):
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

no_outputs = 1
no_inputs = 30 #

#read input
all=(pd.read_csv('WDBC/wdbc.data', sep=",", header=None)).values
inputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[*range(2, 32)])).values
outputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[1])).values


# compute how many examples are for training and testing
training_set_size=math.floor(len(inputs)*0.8) #80%
testing_set_size=len(inputs)-training_set_size #20%

#split in training and test input data
training_inputs= np.asfarray(inputs[0:training_set_size,:])
test_inputs= np.asfarray(inputs[training_set_size:,:])

# change labels malignent to 0.99 and benign to 0.01
outputs=np.where(outputs=='M', 0.99, outputs)
outputs=np.where(outputs=='B', 0.01, outputs)

#split labels in training and test labels
training_outputs= np.asfarray(outputs[0:training_set_size,:])
test_outputs= np.asfarray(outputs[training_set_size:,:])



epochs=100
hidden_nodes=100
L=0.1
bias=0.1



print("-------")

neural_network = NeuralNetwork(no_inputs, hidden_nodes, no_outputs, L, bias)
for epoch in range(epochs):
    neural_network.train(training_inputs, training_outputs)


print("TRAINING SET")
confusion_matrix_training=neural_network.evaluate(training_inputs,training_outputs)
correct,incorrect,accuracy = neural_network.measurements(confusion_matrix_training)
print("Accuracy: ", accuracy)
print("Number of correct instances: ", correct)
print("Number of incorrect instances: ", incorrect)


print("TESTING SET")
confusion_matrix=neural_network.evaluate(test_inputs,test_outputs)
correct,incorrect,accuracy =neural_network.measurements(confusion_matrix)
print("Accuracy: ", accuracy)
print("Number of correct instances: ", correct)
print("Number of incorrect instances: ", incorrect)

