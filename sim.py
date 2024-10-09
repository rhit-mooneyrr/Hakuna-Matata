import numpy as np
import matplotlib.pyplot as plt
import fnn
import ea
from Meerkat import Meerkat
from Hyena import Hyena

class Simulation:
    def __init__(self, env_size=10, max_steps=150, popsize=30):
        self.env_size = env_size
        self.max_steps = max_steps
        self.popsize = popsize
        self.stop_threshold = 0.3

        # Neural network structure for hyena
        self.layers = [2, 4, 2]
        self.genesize = np.sum(np.multiply(self.layers[1:], self.layers[:-1])) + np.sum(self.layers[1:])
        
        # Initialize meerkat at a fixed position
        self.meerkat = Meerkat(5.0, 5.0)

    def fitness_function(self, genotype):
        """Fitness function for microbial GA."""
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(genotype)
        
        total_fitness = 0
        previous_distance = np.linalg.norm(hyena.position - self.meerkat.position)

        for step in range(self.max_steps):
            # Move the hyena and compute new distance
            hyena.move_towards_meerkat(self.meerkat.position)
            new_distance = np.linalg.norm(hyena.position - self.meerkat.position)

            # Reward movement towards the meerkat, penalize moving away
            if new_distance < previous_distance:
                total_fitness += (previous_distance - new_distance) * 10  # Reward for getting closer
            else:
                total_fitness -= 100  # Strong penalty for overshooting

            previous_distance = new_distance

        return total_fitness

    def run_evolution(self):
        """Run the microbial genetic algorithm."""
        ga = ea.MGA(self.fitness_function, self.genesize, self.popsize, 0.7, 0.1, 50 * self.popsize)
        ga.run()

        # Extract the best genotype
        best_genotype = ga.pop[np.argmax(ga.fit)]
        return best_genotype

    def run_simulation(self, best_genotype):
        """Run the simulation with the best evolved hyena."""
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(best_genotype)
        
        positions = [hyena.position.copy()]

        for step in range(self.max_steps):
            hyena.move_towards_meerkat(self.meerkat.position)
            positions.append(hyena.position.copy())

        return np.array(positions)

    def plot_results(self, positions):
        """Plot the path of the hyena and positions of the meerkat."""
        plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], label="Hyena Path", color="blue")
        plt.scatter(self.meerkat.position[0], self.meerkat.position[1], label="Meerkat", color="red", marker="x")
        plt.scatter(positions[0, 0], positions[0, 1], label="Start", color="green", marker="o", s=100)  # Start with larger dot
        plt.scatter(positions[-1, 0], positions[-1, 1], label="End", color="purple", marker="o", s=100)  # End with larger dot
        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.title("Hyena's Path towards Meerkat")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.show()

# Main Code
if __name__ == "__main__":
    sim = Simulation()
    best_genotype = sim.run_evolution()
    positions = sim.run_simulation(best_genotype)
    sim.plot_results(positions)
