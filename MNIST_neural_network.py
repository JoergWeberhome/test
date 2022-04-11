import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

#### data preparation ##############

#TODO: Determine number of input and output neurons
no_outputs = 10
no_inputs = 28*28

#read input
'''
all = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None)).values
inputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[*range(2, 32)])).values
outputs = (pd.read_csv('WDBC/wdbc.data', sep=",", header=None, usecols=[1])).values
'''
training_inputs = np.loadtxt("MNIST_CSV/mnist_test_100.csv",
                             delimiter=',')
# print('-----------------')
# print('Training_inputs:',training_inputs)
# print('Training_inputs:',training_inputs[:,0], 'Länge:',len(training_inputs[:,0]))
test_inputs = np.loadtxt("MNIST_CSV/mnist_train_1000.csv",
                         delimiter=",")

# TODO Extracts the labels from the input files
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

# TODO: Input values should be between 0.01 and 0.99
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