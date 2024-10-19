import numpy as np
import matplotlib.pyplot as plt
from Hyena import Hyena
from Meerkat import Meerkat
import ea

class Simulation:
    def __init__(self, num_meerkats=3, smell_radius=20):
        self.env_size = 100
        self.max_steps = 100
        self.smell_radius = smell_radius
        self.layers = [2, 10, 2]

        self.meerkats = []
        for _ in range(num_meerkats):
            meerkat_x = np.random.uniform(0, self.env_size)
            meerkat_y = np.random.uniform(0, self.env_size)
            self.meerkats.append(Meerkat(meerkat_x, meerkat_y))
        
        self.genesize = (
            self.layers[0] * self.layers[1] +
            self.layers[1] * self.layers[2] +
            self.layers[1] + self.layers[2]
        )

        self.popsize = 50
    
    def fitness_function(self, genotype):
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(genotype)

        total_fitness = 0

        for step in range(self.max_steps):
            closest_meerkat, closest_distance = hyena.detect_meerkat(self.meerkats)

            if closest_meerkat:
                hyena.move_towards_meerkat(closest_meerkat.position)
                new_distance = hyena.calculate_distance(closest_meerkat.position)

                if new_distance < closest_distance:
                    total_fitness += (closest_distance - new_distance) * 10
                else:
                    total_fitness -= 50
                
                closest_distance = new_distance

                if new_distance < 5:
                    total_fitness += 1000
                    break
            else:
                total_fitness -= 10
            
        return total_fitness
    
    def find_closest_meerkat(self, hyena):
        closest_meerkat = None
        closest_distance = float('inf')
        for meerkat in self.meerkats:
            distance = hyena.calculate_distance(meerkat.position)
            if distance < closest_distance:
                closest_distance = distance
                closest_meerkat = meerkat
        return closest_meerkat
    
    def run_evolution(self):
        ga = ea.MGA(self.fitness_function, self.genesize, self.popsize, 0.7, 0.1, 50*self.popsize)
        ga.run()

        best_genotype = ga.pop[np.argmax(ga.fit)]
        return best_genotype
    
    def run_simulation(self, best_genotype):
        hyena = Hyena(self.layers, self.env_size)
        hyena.set_genotype(best_genotype)

        positions = [hyena.position.copy()]

        for step in range(self.max_steps):
            closest_meerkat = self.find_closest_meerkat(hyena)
            hyena.move_towards_meerkat(closest_meerkat.position)
            positions.append(hyena.position.copy())

        return np.array(positions)
    
    def plot_results(self, positions):
        plt.figure()
        plt.plot(positions[:,0], positions[:,1], label = "Hyena Path", color="blue")

        hyena_final_position = positions[-1]

        for i, meerkat in enumerate(self.meerkats):
            meerkat_position = meerkat.position
            distance_to_hyena = np.linalg.norm(hyena_final_position - meerkat_position)

            meerkat_circle = plt.Circle(meerkat_position, self.smell_radius, color='gray', alpha=0.3, linestyle='--')
            plt.gca().add_artist(meerkat_circle)

            if distance_to_hyena < 5:
                plt.scatter(meerkat_position[0], meerkat_position[1], label=f"Merkat {i+1} (Reached)",
                            color="orange", marker="x", s=200, linewidths=2)
            else:
                plt.scatter(meerkat_position[0], meerkat_position[1], label=f"Meerkat {i+1}",
                            color="red", marker="x", s=100)
        
        plt.scatter(positions[0,0], positions[0,1], label="Start", color="green", marker="o", s=100)
        plt.scatter(positions[-1,0], positions[-1,1], label="End", color="purple", marker="o", s=100)
        plt.xlim(0, self.env_size)
        plt.ylim(0, self.env_size)
        plt.title("Hyena's Path towards Meerkats")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    sim = Simulation(num_meerkats=3)
    best_genotype = sim.run_evolution()
    positions = sim.run_simulation(best_genotype)
    sim.plot_results(positions)