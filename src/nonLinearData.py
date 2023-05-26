'''
In this program we create a non-linear set of data points.
'''

import numpy as np
import nnfs.datasets as nnfs
import matplotlib.pyplot as plt

np.random.seed(0)

def main():
    
    class DenseLayer:

        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases

    X, y = nnfs.spiral_data(samples=100,  classes=3)
    print("Shape of X: \n", X.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()
    plt.close("all")

    dense_1 = DenseLayer(2, 3)
    dense_1.forward(X)
    print(dense_1.output[:5])


if __name__ == "__main__":
    main()