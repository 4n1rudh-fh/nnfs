'''
- This is a beginner's program to create a NN. 
- The outputs, weights and biases are coming from 3 neurons in previous layer and are input to the single neuron we create in this program.
'''

# Inputs to 'the' single neuron
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

# output from 'the' single neuron
output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + bias
print("The output is: ", output)