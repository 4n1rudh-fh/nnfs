'''
In this program:
- We created multiple input data batches.
- We created 2 layers of neurons. One with 4 neurons and another with 3.
- As can be seen, creating layers this way is getting more cumbersome.
- For next step, checkout 'creatingObjectLayers.py'
'''

import numpy as np

def main():

    # first layer

    # 3x4 matrix
    inputs = np.array([[1, 2, 3, 2.5],
                       [2.0, 5.0, -1.0, 2.0],
                       [-1.5, 2.7, 3.3, -0.8]])
    # print(inputs.shape)

    # 3x4 matrix
    weights = np.array([[0.2, 0.8, -0.5, 1.0],
                        [0.5, -0.91, 0.26, -0.5],
                        [-0.26, -0.27, 0.17, 0.87]])
    # print(weights.shape)

    # 3x1 vector
    biases = np.array([2, 3, 0.5])
    print(biases.shape)

    # second layer

    weights_2 = np.array([[0.1, -0.14, 0.5],
                        [-0.5, 0.12, -0.33],
                        [-0.44, 0.73, -0.13]])
    
    biases_2 = np.array([-1, 2, -0.5])
    
    layer1_outputs = np.dot(inputs, np.transpose(weights)) + biases
    layer2_outputs = np.dot(layer1_outputs, np.transpose(weights_2)) + biases_2

    print("Layer 1 Outputs: \n", layer1_outputs)
    print("\n Layer 2 Outputs: \n", layer2_outputs)


if __name__ == "__main__":
    main()