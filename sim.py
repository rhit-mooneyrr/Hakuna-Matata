import numpy as np
import matplotlib.pyplot as plt
from Hyena import Hyena
from Meerkat import Meerkat
import ea

class Simulation:
    def __init__(self, num_meerkats=3):
        self.env_size = 100
        self.max_steps = 1000
        self.layers = [1, 10, 2]  # 1 input (smell_strength), hidden layer with 10 units, 2 outputs (delta x, delta y)
        
        # Initialize multiple meerkats with a smell radius
        self.meerkats = []
        for _ in range(num_meerkats):
            meerkat_x = np.random.uniform(0, self.env_size)
            meerkat_y = np.random.uniform(0, self.env_size)
            smell_radius = 20  # Define the smell radius
            self.meerkats.append(Meerkat(meerkat_x, meerkat_y, smell_radius))
        
        # Calculate total number of weights and biases for the neural network
        self.genesize = (
            self.layers[0] * self.layers[1] +   # Input to hidden layer weights
            self.layers[1] * self.layers[2] +   # Hidden to output layer weights
            self.layers[1] +                    # Hidden layer biases
            self.layers[2]                      # Output layer biases
        )

        self.popsize = 50

    def fitness_function(self, genotype):
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(genotype)

        total_fitness = 0
        previous_smell_strength = 0  # To track changes in smell strength

        for step in range(self.max_steps):
            # Find the meerkat with the strongest smell
            target_meerkat, smell_strength = self.find_strongest_smell(hyena)

            if smell_strength > 0:
                # Calculate direction towards target meerkat
                direction = target_meerkat.position - hyena.position
                target_direction = direction / np.linalg.norm(direction)
            else:
                target_direction = np.zeros(2)

            hyena.move(smell_strength, target_direction)

            # Update fitness
            if smell_strength > previous_smell_strength:
                total_fitness += (smell_strength - previous_smell_strength) * 200
            elif smell_strength < previous_smell_strength:
                total_fitness -= (previous_smell_strength - smell_strength) * 200
            else:
                total_fitness -= 1  # Small penalty for no improvement

            if smell_strength == 0 and previous_smell_strength == 0:
                # Encourage exploration
                total_fitness += 0.2  # Small reward for moving

            previous_smell_strength = smell_strength

            # Large reward for reaching a meerkat
            if target_meerkat is not None:
                distance = hyena.calculate_distance(target_meerkat.position)
                if distance < 5:
                    total_fitness += 1000
                    break  # Stop if the hyena is close enough to a meerkat

        return total_fitness

    def find_strongest_smell(self, hyena):
        """Find the meerkat with the strongest smell."""
        strongest_smell = 0
        target_meerkat = None
        for meerkat in self.meerkats:
            distance = hyena.calculate_distance(meerkat.position)
            if distance <= meerkat.smell_radius:
                smell_strength = 1.0 / (1.0 + distance)
                if smell_strength > strongest_smell:
                    strongest_smell = smell_strength
                    target_meerkat = meerkat
                elif abs(smell_strength - strongest_smell) < 0.01:
                    # If smells are approximately equal, randomly choose
                    if np.random.rand() > 0.5:
                        strongest_smell = smell_strength
                        target_meerkat = meerkat
        return target_meerkat, strongest_smell

    def run_evolution(self):
        """Run the microbial genetic algorithm."""
        ga = ea.MGA(self.fitness_function, self.genesize, self.popsize, 0.7, 0.1, 50 * self.popsize)
        ga.run()

        # Extract the best genotype
        best_genotype = ga.pop[np.argmax(ga.fit)]
        # Plot the fitness over time
        ga.showFitness()
        return best_genotype

    def run_simulation(self, best_genotype):
        """Run the simulation with the best evolved hyena."""
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(best_genotype)
        
        positions = [hyena.position.copy()]

        previous_smell_strength = 0

        for step in range(self.max_steps):
            target_meerkat, smell_strength = self.find_strongest_smell(hyena)

            if smell_strength > 0:
                # Calculate direction towards target meerkat
                direction = target_meerkat.position - hyena.position
                target_direction = direction / np.linalg.norm(direction)
            else:
                target_direction = np.zeros(2)

            hyena.move(smell_strength, target_direction)
            positions.append(hyena.position.copy())

            # Stop if hyena reaches a meerkat
            if target_meerkat is not None:
                distance = hyena.calculate_distance(target_meerkat.position)
                if distance < 5:
                    break

            previous_smell_strength = smell_strength

        return np.array(positions)

    def plot_results(self, positions):
        """Plot the path of the hyena and positions of the meerkats."""
        plt.figure(figsize=(10, 8))
        plt.plot(positions[:, 0], positions[:, 1], label="Hyena Path", color="blue")
        
        hyena_final_position = positions[-1]

        for i, meerkat in enumerate(self.meerkats):
            meerkat_position = meerkat.position
            distance_to_hyena = np.linalg.norm(hyena_final_position - meerkat_position)
            
            # Plot the smell radius
            circle = plt.Circle((meerkat_position[0], meerkat_position[1]), meerkat.smell_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
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
        plt.grid(True)
        plt.show()

# Main Code
if __name__ == "__main__":
    sim = Simulation(num_meerkats=3)  # You can set the number of meerkats here
    best_genotype = sim.run_evolution()
    positions = sim.run_simulation(best_genotype)
    sim.plot_results(positions)
