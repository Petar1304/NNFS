import numpy as np 
# getting data
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

np.random.seed(0)

# input batch_size = 3, input dim = (1, 4)
X = np.array([[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]])


class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.rand(n_inputs, n_neurons) # transpose already done with changing order of n_inputs and n_neurons
    self.biases = np.zeros((1, n_neurons)) # initialize with zero

  # forward pass
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
    

class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

# SOFTMAX
class Activation_Softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities



X, y = spiral_data(samples=10, classes=3)
activation1 = Activation_ReLU()
 
# layer1 = Layer_Dense(2, 10) # first parameter is how many inputs do we have, second could be anything
# layer2 = Layer_Dense(10, 2) # output from layer1 is input, output could be anything


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)






