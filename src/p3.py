'''
Program to create an output layer (the final one) with 3 neurons which is fully connected to 4 neurons in the hidden layer. The hidden layer is present right before the output layer. 
'''

# The 4 neurons in the hidden layer
inputs = [1, 2, 3, 2.5]

# The weights for each neuron in the hidden layer. There are 3 lists for each output neuron.
# To imagine: look from a single output layer neuron's perspective
weights_1 = [0.2, 0.8, -0.5, 1.0]
weights_2 = [0.5, -0.91, 0.26, -0.5]
weights_3 = [-0.26, -0.27, 0.17, 0.87]

# each output neuron is still going to have only a single bias number. Because there are 3 output neurons, there are 3 biases.
bias_1 = 2
bias_2 = 3
bias_3 = 0.5

# output as a list for each single neuron in output layer
output = [
        ((inputs[0] * weights_1[0]) + (inputs[1] * weights_1[1]) + (inputs[2] * weights_1[2]) + (inputs[3] * weights_1[3])+ bias_1), 
        ((inputs[0] * weights_2[0]) + (inputs[1] * weights_2[1]) + (inputs[2] * weights_2[2]) + (inputs[3] * weights_2[3])+ bias_2), 
        ((inputs[0] * weights_3[0]) + (inputs[1] * weights_3[1]) + (inputs[2] * weights_3[2]) + (inputs[3] * weights_3[3])+ bias_3)
        ]

print("The output is: ", output)