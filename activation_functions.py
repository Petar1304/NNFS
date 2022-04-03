# SOFTMAX
# mostly used to get probability distribution for the output layer
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.0026]]

exp_values = np.exp(layer_outputs)
# norm_values = exp_values / sum(exp_values)

# print(np.sum(layer_outputs)) # = 18.148

print(np.sum(layer_outputs, axis=1, keepdims=True))
# axis=0 -> sum of columns
# axis=1 -> sum of rows

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)














