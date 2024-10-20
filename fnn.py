import numpy as np

class FNN:
    def __init__(self, layers):
        """Initialize the neural network with a list of layers."""
        self.layers = layers
        self.units_per_layer = layers
        self.weights = []
        self.biases = []
        self.weightrange = 1.0
    
    def setParams(self, params):
        """Set the neural network parameters (weights and biases)."""
        self.weights = []
        self.biases = []
        start = 0
        
        for l in range(len(self.units_per_layer) - 1):
            # Set weights for the connections between layers
            end = start + self.units_per_layer[l] * self.units_per_layer[l + 1]
            self.weights.append((params[start:end] * self.weightrange).reshape(self.units_per_layer[l], self.units_per_layer[l + 1]))
            start = end
            
            # Set biases for the current layer
            end = start + self.units_per_layer[l + 1]
            self.biases.append((params[start:end] * self.weightrange))
            start = end

    def forward(self, inputs):
        """Perform a forward pass through the network."""
        x = inputs
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i]) + self.biases[i]  # Weighted sum + bias
            x = self.sigmoid(x)  # Apply activation function
        return x

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def getParams(self):
        """Get the neural network parameters as a flat list (for use in evolution)."""
        params = []
        for weight in self.weights:
            params.extend(weight.flatten())
        for bias in self.biases:
            params.extend(bias.flatten())
        return np.array(params)
