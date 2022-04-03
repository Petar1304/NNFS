import numpy as np 

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
    

 
layer1 = Layer_Dense(4, 10) # first parameter is how many inputs do we have, second could be anything
layer2 = Layer_Dense(10, 2) # output from layer1 is input, output could be anything

# forward pass throught layers
layer1.forward(X)

layer2.forward(layer1.output)

print('Layer 1 output:\n', layer1.output)
print('Layer 2 output:\n', layer2.output)



