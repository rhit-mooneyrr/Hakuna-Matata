import numpy as np
import fnn  # Assuming FNN is part of the fnn.py module

class Hyena:
    def __init__(self, layers, env_size, smell_radius=20):
        self.env_size = env_size
        self.position = np.random.uniform(0, env_size, 2)  # Random initial position
        self.brain = fnn.FNN(layers)  # Initialize neural network
        self.previous_movement = np.random.uniform(-1, 1, 2)  # Initial random direction

    def set_genotype(self, genotype):
        self.brain.setParams(genotype)

    def move(self, smell_strength, target_direction):
        """Move the hyena based on smell strength."""
        if smell_strength > 0:
            input_data = np.array([smell_strength])
            movement = self.brain.forward(input_data)
            # Map output from [0,1] to [-2,2] for movement
            movement = (movement - 0.5) * 4.0
            # Combine movement with target direction
            movement = movement * 0.5 + target_direction * 2.0  # Strongly bias towards target
        else:
            # Random movement with more frequent direction changes
            if np.random.rand() < 0.5:
                # Continue in the same direction
                movement = self.previous_movement
            else:
                # Change direction
                movement = np.random.uniform(-2, 2, 2)
            self.previous_movement = movement  # Update previous movement

        self.position += movement
        self.position = np.clip(self.position, 0, self.env_size)
        self.previous_movement = movement  # Update previous movement

    def calculate_distance(self, target_position):
        return np.linalg.norm(self.position - target_position)