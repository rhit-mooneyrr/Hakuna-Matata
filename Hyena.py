import numpy as np
import matplotlib.pyplot as plt
import fnn
import ea

class Hyena:
    def __init__(self, layers, env_size, smell_radius=20):
        self.env_size = env_size
        self.position = np.random.uniform(0, env_size, 2)
        self.brain = fnn.FNN(layers)
        self.smell_radius = smell_radius
    
    def set_genotype(self, genotype):
        self.brain.setParams(genotype)
    
    def detect_meerkat(self, meerkats):
        closest_meerkat = None
        closest_distance = float('inf')
    
        for meerkat in meerkats:
            distance = np.linalg.norm(self.position - meerkat.position)
            if distance < self.smell_radius and distance < closest_distance:
                closest_meerkat = meerkat
                closest_distance = distance
            
        return closest_meerkat, closest_distance
    
    def move_towards_meerkat(self, meerkat_position):
        distance = np.linalg.norm(self.position - meerkat_position)
        smell_strength = 1.0 / (1.0 + distance)

        input_data = np.array([distance, smell_strength])
        movement = self.brain.forward(input_data)

        movement = np.squeeze(movement)
        if np.linalg.norm(movement) > 0:
            direction = meerkat_position - self.position
            movement = (direction / np.linalg.norm(direction)) * 2.0
        
        if np.linalg.norm(movement) < 0.05:
            movement = np.sign(movement)
        
        self.position += movement
        self.position = np.clip(self.position, 0, self.env_size)

    def calculate_distance(self, target_position):
        return np.linalg.norm(self.position - target_position)