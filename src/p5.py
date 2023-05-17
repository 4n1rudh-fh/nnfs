'''
Program to create an output layer (the final one) with 3 neurons which is fully connected to 4 neurons in the hidden layer. The hidden layer is present right before the output layer.

Edit: We use numpy here.
'''

import numpy as np

def main():

    # 4x1 vector
    inputs = [1, 2, 3, 2.5]

    # 3x4 vector
    weights = [[0.2, 0.8, -0.5, 1.0],
              [0.5, -0.91, 0.26, -0.5],
              [-0.26, -0.27, 0.17, 0.87]]

    # 3x1 vector
    biases = [2, 3, 0.5]

    # create an empty output list
    layer_outputs = np.dot(weights, inputs) + biases
    print("The output is: ", layer_outputs)


if __name__ == "__main__":
    main()