import matplotlib.pyplot as plt


class Graph:
    """
    class for graph ploting that holds the values from the differnt runs of the genetic algorithm in order to represent the avrage values from the 
    """

    def __init__(self, generations):
        self.all_best_fitness = [0] * generations
        self.all_worst_fitness = [0] * generations
        self.all_average_fitness =[0] * generations
        self.num_of_gen = [i for i in range(generations)]
        self.best_fitness = 0
        
    def plot_graph(self,pop, gen, rate):
       
        # x axis values
        x = self.num_of_gen
        # corresponding y axis values
        y1 = self.all_best_fitness
        y2 = self.all_worst_fitness
        y3 = self.all_average_fitness

        # plotting the points
        plt.plot(x, y1, color='blue', label="Best Fitness")
        plt.plot(x, y2, color='red', label="Worst Fitness")
        plt.plot(x, y3, color='green', label="Average Fitness")

        # naming the x axis
        plt.xlabel('Number of generations')
        # naming the y axis
        plt.ylabel('Fitness values')

        # giving a title to my graph
        # plt.title('Fitness/Generations')
        plt.title(f"Population: {pop} Generations: {gen} Mutation Rate: {rate}")

        # show a legend on the plot
        plt.legend()

        # function to show the plot
        plt.show()