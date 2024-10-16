import numpy as np
import matplotlib.pyplot as plt
from Hyena import Hyena
from Meerkat import Meerkat
import ea

class Simulation:
    def __init__(self, num_meerkats=3):
        self.env_size = 100
        self.max_steps = 100
        self.layers = [2, 10, 2]  # Example: 2 inputs (x, y), hidden layer with 10 units, 2 outputs (delta x, delta y)
        
        # Initialize multiple meerkats
        self.meerkats = []
        for _ in range(num_meerkats):
            meerkat_x = np.random.uniform(0, self.env_size)
            meerkat_y = np.random.uniform(0, self.env_size)
            self.meerkats.append(Meerkat(meerkat_x, meerkat_y))
        
        # Calculate total number of weights and biases for the neural network
        self.genesize = (
            self.layers[0] * self.layers[1] +   # Input to hidden layer weights
            self.layers[1] * self.layers[2] +   # Hidden to output layer weights
            self.layers[1] +                   # Hidden layer biases
            self.layers[2]                     # Output layer biases
        )

        self.popsize = 50

    def fitness_function(self, genotype):
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(genotype)

        total_fitness = 0

        # Track the closest meerkat distance
        closest_meerkat = None
        closest_distance = float('inf')

        for step in range(self.max_steps):
            # Find the closest meerkat
            closest_meerkat = self.find_closest_meerkat(hyena)

            # Move the hyena towards the closest meerkat
            hyena.move_towards_meerkat(closest_meerkat.position)
            new_distance = hyena.calculate_distance(closest_meerkat.position)

            # Reward for getting closer, penalty for moving away
            if new_distance < closest_distance:
                total_fitness += (closest_distance - new_distance) * 10
            else:
                total_fitness -= 50

            closest_distance = new_distance

            # Large reward for reaching a meerkat
            if new_distance < 5:
                total_fitness += 1000
                break  # Stop if the hyena is close enough to a meerkat

        return total_fitness

    def find_closest_meerkat(self, hyena):
        """Find the meerkat closest to the hyena."""
        closest_meerkat = None
        closest_distance = float('inf')
        for meerkat in self.meerkats:
            distance = hyena.calculate_distance(meerkat.position)
            if distance < closest_distance:
                closest_distance = distance
                closest_meerkat = meerkat
        return closest_meerkat

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
            closest_meerkat = self.find_closest_meerkat(hyena)
            hyena.move_towards_meerkat(closest_meerkat.position)
            positions.append(hyena.position.copy())

        return np.array(positions)

    def plot_results(self, positions):
        """Plot the path of the hyena and positions of the meerkats."""
        plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], label="Hyena Path", color="blue")
        
        # Check if the hyena's final position overlaps with any meerkat
        hyena_final_position = positions[-1]

        for i, meerkat in enumerate(self.meerkats):
            meerkat_position = meerkat.position
            distance_to_hyena = np.linalg.norm(hyena_final_position - meerkat_position)
            
            if distance_to_hyena < 5:  # If the final position is within 5 units of the meerkat
                plt.scatter(meerkat_position[0], meerkat_position[1], label=f"Meerkat {i+1} (Reached)", 
                            color="orange", marker="x", s=200, edgecolor="black", linewidths=2)  # Highlight meerkat
            else:
                plt.scatter(meerkat_position[0], meerkat_position[1], label=f"Meerkat {i+1}", 
                            color="red", marker="x", s=100)  # Regular size

        plt.scatter(positions[0, 0], positions[0, 1], label="Start", color="green", marker="o", s=100)
        plt.scatter(positions[-1, 0], positions[-1, 1], label="End", color="purple", marker="o", s=100)
        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.title("Hyena's Path towards Meerkats")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.show()

# Main Code
if __name__ == "__main__":
    sim = Simulation(num_meerkats=3)  # You can set the number of meerkats here
    best_genotype = sim.run_evolution()
    positions = sim.run_simulation(best_genotype)
    sim.plot_results(positions)
