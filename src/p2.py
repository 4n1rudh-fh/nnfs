# Inputs to 'the' single final output neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2 # the output neuron is still going to have only a single bias number

# output from 'the' single neuron
output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + (inputs[3] * weights[3])+ bias
print("The output is: ", output)