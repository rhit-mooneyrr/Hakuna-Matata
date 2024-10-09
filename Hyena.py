import numpy as np

class Hyena():

    def __init__(self, target, fitnessfunction, popsize, mutation_rate, generations):
        self.target = np.array(target)
        self.fitnessfunction = fitnessfunction
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.generations = generations

    def fitness(self, genome):
        position = np.zeros(2)
        position += genome
        distance = np.linalg.norm(self.target - position)
        return 1 / (1 + distance)
    