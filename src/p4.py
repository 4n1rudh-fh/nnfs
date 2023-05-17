'''
Program to create an output layer (the final one) with 3 neurons which is fully connected to 4 neurons in the hidden layer. The hidden layer is present right before the output layer.

Edit: We create output layer using pure python and loops.
'''

import numpy as np

def main():
    # The 4 neurons in the hidden layer
    inputs = [1, 2, 3, 2.5]

    # To imagine: look from a single output layer neuron's perspective
    # All weights from p3.py have been put inside a single list of lists variable.
    weights = [[0.2, 0.8, -0.5, 1.0],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]

    # each output neuron is still going to have only a single bias number. Because there are 3 output neurons, there are 3 biases.
    biases = [2, 3, 0.5]

    # create an empty output list
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
        
    print("The output is: ", layer_outputs)

if __name__ == "__main__":
    main()