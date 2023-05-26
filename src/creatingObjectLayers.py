'''
In this program, we define two hidden layers.
'''

import numpy as np

np.random.seed(0)

def main():

    # convention to declare input data as capital X.
    X = np.array([[1, 2, 3, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8]])
    
    class LayerDense:

        def __init__(self, n_inputs, n_neurons):
            # to have values between 1 and -1 we multiply with 0.1
            # also, no transpose for weight matrix with this order
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

    layer_1 = LayerDense(4, 3)
    layer_2 = LayerDense(3, 3)

    layer_1.forward(X)
    layer_2.forward(layer_1.output)
    print("Layer 1 output: \n", layer_1.output)
    print("Layer 2 output: \n", layer_2.output)
    

if __name__ == "__main__":
    main()