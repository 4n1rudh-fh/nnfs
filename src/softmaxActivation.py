'''
Illustrates the need for a different activation function for output layer.
'''
import numpy as np
import math
import nnfs
import nnfs.datasets as nnfs_data

nnfs.init()

def main():

    # creating layer
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

    # creating ReLU
    class Activation_ReLU:
        def forward(self, inputs):
            self.output = np.maximum(0, inputs)

    # creating Softmax
    class Activation_Softmax:
        def forward(self, inputs):
            '''Softmax function = exponentiation, subtracting max & normalization'''
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities
    
    # assuming these are outputs from hidden layer before the o/p layer
    X, y = nnfs_data.spiral_data(samples=100, classes=3)
    
    dense_1 = Layer_Dense(2, 3)
    activation_1 = Activation_ReLU()

    dense_2 = Layer_Dense(3, 3)
    activation_2 = Activation_Softmax()

    dense_1.forward(X)
    activation_1.forward(dense_1.output)

    dense_2.forward(activation_1.output)
    activation_2.forward(dense_2.output)

    '''Each datapoint has 2 values i.e. x, y. These values go through the network and then "probable" class, the datapoint belongs to, is determined.'''
    print(activation_2.output[:5])

    
if __name__ == "__main__":
    main()