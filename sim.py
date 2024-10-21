import numpy as np
import matplotlib.pyplot as plt
from Hyena import Hyena
from Meerkat import Meerkat
import ea

class Simulation:
    def __init__(self, num_hyenas=3, num_meerkats=3):
        self.env_size = 100
        self.max_steps = 100
        self.layers = [1, 4, 2]  # Neural network architecture
        self.num_hyenas = num_hyenas
        self.num_meerkats = num_meerkats

        # Initialize multiple meerkats with a smell radius
        self.meerkats = []
        for _ in range(self.num_meerkats):
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

        self.popsize = 50  # Population size for the evolutionary algorithm

    def fitness_function(self, genotype):
        total_fitness = 0

        # Simulate multiple hyenas with the same genotype
        for _ in range(self.num_hyenas):
            hyena = Hyena(self.layers, self.env_size)
            hyena.set_genotype(genotype)
            hyena.position = np.random.uniform(0, self.env_size, 2)  # Random initial position
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

        # Average fitness over the number of hyenas
        average_fitness = total_fitness / self.num_hyenas
        return average_fitness

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

        # Plot the fitness over time
        ga.showFitness()

        # Extract the best genotype
        best_genotype = ga.pop[np.argmax(ga.fit)]
        return best_genotype

    def run_simulation(self, best_genotype):
        """Run the simulation with the best evolved hyenas."""
        hyenas = []
        positions = []

        # Initialize hyenas with the best genotype
        for _ in range(self.num_hyenas):
            hyena = Hyena(self.layers, self.env_size)
            hyena.set_genotype(best_genotype)
            hyena.position = np.random.uniform(0, self.env_size, 2)  # Random initial position
            hyenas.append(hyena)
            positions.append([hyena.position.copy()])

        # Simulate movement for each hyena
        for step in range(self.max_steps):
            for idx, hyena in enumerate(hyenas):
                target_meerkat, smell_strength = self.find_strongest_smell(hyena)

                if smell_strength > 0:
                    # Calculate direction towards target meerkat
                    direction = target_meerkat.position - hyena.position
                    target_direction = direction / np.linalg.norm(direction)
                else:
                    target_direction = np.zeros(2)

                hyena.move(smell_strength, target_direction)
                positions[idx].append(hyena.position.copy())

        # Convert positions to numpy arrays
        positions = [np.array(pos_list) for pos_list in positions]
        return positions

    def plot_results(self, positions):
        """Plot the paths of the hyenas and positions of the meerkats."""
        plt.figure(figsize=(10, 8))

        # Define colors and markers for hyenas
        colors = ['blue', 'green', 'purple']  # Distinct colors for 3 hyenas
        markers = ['o', 's', '^']  # Different markers for hyenas

        # Plot hyenas' paths
        for idx, hyena_positions in enumerate(positions):
            hyena_color = colors[idx % len(colors)]
            hyena_marker = markers[idx % len(markers)]
            plt.plot(hyena_positions[:, 0], hyena_positions[:, 1],
                     label=f"Hyena {idx+1} Path", color=hyena_color, linestyle='-', linewidth=2)
            plt.scatter(hyena_positions[0, 0], hyena_positions[0, 1],
                        color=hyena_color, marker=hyena_marker, s=100, edgecolors='black',
                        label=f"Hyena {idx+1} Start")
            plt.scatter(hyena_positions[-1, 0], hyena_positions[-1, 1],
                        color=hyena_color, marker=hyena_marker, s=100, edgecolors='black', facecolors='none',
                        label=f"Hyena {idx+1} End")

        # Plot meerkats
        for i, meerkat in enumerate(self.meerkats):
            meerkat_position = meerkat.position

            # Plot the smell radius
            circle = plt.Circle((meerkat_position[0], meerkat_position[1]),
                                meerkat.smell_radius, color='gray', fill=False,
                                linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)

            plt.scatter(meerkat_position[0], meerkat_position[1],
                        color="red", marker="D", s=150, edgecolors='black',
                        label=f"Meerkat {i+1}")

        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.title("Hyenas' Paths towards Meerkats")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.grid(True)

        # Create custom legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Remove duplicate labels while preserving order
        from collections import OrderedDict
        by_label = OrderedDict()
        for handle, label in zip(handles, labels):
            if label not in by_label:
                by_label[label] = handle

        # Adjust legend placement and formatting
        plt.legend(by_label.values(), by_label.keys(),
                   loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   fancybox=True, shadow=True, ncol=3, fontsize='medium')

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for legend
        plt.show()

# Main Code
if __name__ == "__main__":
    sim = Simulation(num_hyenas=3, num_meerkats=3)
    best_genotype = sim.run_evolution()
    positions = sim.run_simulation(best_genotype)
    sim.plot_results(positions)
