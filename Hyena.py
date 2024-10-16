import numpy as np
import matplotlib as plt
import fnn
import numpy as np
import fnn  # Assuming FNN is part of the fnn.py module

class Hyena:
    def __init__(self, layers, env_size):
        self.env_size = env_size
        self.position = np.random.uniform(0, env_size, 2)  # Random initial position
        self.brain = fnn.FNN(layers)  # Initialize neural network

    def set_genotype(self, genotype):
        """Set the neural network parameters."""
        self.brain.setParams(genotype)

    def move_towards_meerkat(self, meerkat_position):
        """Move the hyena based on its neural network output."""
        distance = np.linalg.norm(self.position - meerkat_position)
        smell_strength = 1.0 / (1.0 + distance)  # Smell inversely proportional to distance

        # Neural network input: distance and smell
        input_data = np.array([distance, smell_strength])
        movement = self.brain.forward(input_data)

        # Normalize and scale the movement towards the meerkat
        movement = np.squeeze(movement)
        if np.linalg.norm(movement) > 0:
            direction = meerkat_position - self.position
            movement = (direction / np.linalg.norm(direction)) * 2.0  # Reduce scaling for smoother tracking

        # Add a minimum movement threshold to avoid getting stuck
        if np.linalg.norm(movement) < 0.05:
            movement = (np.sign(movement) * 0.05)

        # Update position and ensure it's within bounds
        self.position += movement
        self.position = np.clip(self.position, 0, self.env_size)

    def calculate_distance(self, target_position):
        """Calculate the Euclidean distance between the hyena and the target (meerkat)."""
        return np.linalg.norm(self.position - target_position)
