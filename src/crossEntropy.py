'''In this program, we implement the cross-entropy loss in our NN.'''

import numpy as np
import nnfs
import nnfs.datasets as nnfs_data

nnfs.init()

def main():
    class Layer_Dense:
        def __init__(self, n_inputs, n_neurons):
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        def forward(self, inputs):
            self.output = np.dot(inputs, self.weights) + self.biases
    
    class Activation_ReLU:
        def forward(self, inputs):
            self.output = np.maximum(0, inputs)
    
    class Activation_Softmax:
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probability = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probability

    # common loss class
    class Loss:
        # output from model
        # y = intended target values
        def calculate(self, output, y):
            # forward method varies depending on type of loss we use
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss
    
    class Loss_Categorical_Cross_Entropy(Loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)

            # 1e-7 to prevent 0 in output and hence infinity
            # 1-1e-7 to be unbiased
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape == 2):
                correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
            negative_log_likelihood = -np.log(correct_confidences)
            return negative_log_likelihood

    # batch input data
    X, y = nnfs_data.spiral_data(samples=100, classes=3)

    # creating first layer
    layer_1 = Layer_Dense(2, 3)
    layer_1.forward(X)

    activation_1 = Activation_ReLU()
    activation_1.forward(layer_1.output)
    layer_2 = Layer_Dense(3, 3)
    layer_2.forward(activation_1.output)
    
    activation_2 = Activation_Softmax()
    activation_2.forward(layer_2.output)

    # print(activation_2.output[:5])

    loss_function = Loss_Categorical_Cross_Entropy()
    loss = loss_function.calculate(activation_2.output, y)
    print("Loss:", loss)


if __name__ == "__main__":
    main()