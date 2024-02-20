
import numpy as np


class Generation:
    """
    this class is used to store the information of a generation of a genetic algorithm.
    later it will be used to plot the best, worst and average fitness of each generation.
    """
    def __init__(self, generation_num: int, fitness: list[float]):
        self.generation_num: int = generation_num
        self.fitness: list[float] = fitness
        self.best_fitness: float = self.set_best_fitness()
        self.worst_fitness: float = self.set_worst_fitness()
        self.average_fitness: float = self.set_average_fitness()

    def set_best_fitness(self):
        return max(self.fitness)

    def set_worst_fitness(self):
       return min(self.fitness)

    def set_average_fitness(self):
        return np.mean(self.fitness)
    
    def __str__(self) -> str:
        return f"generation_num: {self.generation_num}:\nBest Fitness: {self.best_fitness}\nWorst Fitness: {self.worst_fitness}\nAverage Fitness: {self.average_fitness}\n"