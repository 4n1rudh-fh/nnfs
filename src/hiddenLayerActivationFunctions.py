# importing libraries
import numpy as np
import nnfs
import nnfs.datasets as nnfs_data
import matplotlib.pyplot as plt

# initialising nnfs for seed
nnfs.init()

# main
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
    
    # importing data from library
    X, y = nnfs_data.spiral_data(samples=100, classes=3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
    plt.show()
    plt.close("all")

    # creating a layer
    layer_1 = Layer_Dense(2, 5)

    # creating activation function
    activation_1 = Activation_ReLU()

    # output from the created layer
    layer_1.forward(X)

    # output from ReLU
    activation_1.forward(layer_1.output)

    print("Layer 1 Output: \n", layer_1.output[:10])
    print("ReLU Output: \n", activation_1.output[:10])

    '''Demonstrates ReLU working
    activation_2 = Activation_ReLU()
    x = np.array([[-1, 2, -3, 4, -5, -6, 7, 8, 9, -10]])
    activation_2.forward(x)
    print(activation_2.output)
    '''

if __name__ == "__main__":
    main()