import math

softmax_output = [0.7, 0.2, 0.1]
target_output = [1, 0, 0]

# this whole term is categorical cross-entropy loss
# but because of one-hot encoding only the first term counts as rest are zero
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(loss)

# the loss above reduces to
loss_new = -math.log(softmax_output[0])
print(loss_new)

# assuming these are predictions for a class for three different images
shit_prediction = [0.1, 0.5, 0.7]

# the loss becomes higher for shittier predictions
print(-math.log(shit_prediction[0]))
print(-math.log(shit_prediction[1]))
print(-math.log(shit_prediction[2]))