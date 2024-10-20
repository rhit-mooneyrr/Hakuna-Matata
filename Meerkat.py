import numpy as np

class Meerkat:
    def __init__(self, x, y, smell_radius=20):
        self.position = np.array([x, y])
        self.smell_radius = smell_radius
