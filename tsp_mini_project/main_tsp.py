#!/usr/bin/env python3


from math import inf
from graph import Graph
from tspparse   import read_tsp_file
from algorithms import calc_christofides_algorithm_tour, calc_genetic_algorithm_tour, calc_held_karp_tour, calc_naive_tour, calc_nearest_neighbor_tour

def calculate_average(cur_tour_values):
    return sum(cur_tour_values) / len(cur_tour_values)

def calc():
    file_path = "results.txt"  

    cur_tour_values = []
    population = [100,300,500]
    generations = [100,500,1000]
    mutation_rate = [0.05,0.09,0.1]
    experiments = []
    for pop in population:
        for gen in generations:
            for rate in mutation_rate:
                experiments.append([pop,gen,rate])
    idx = 0
    f = open ("average_results.txt", "w")
    with open(file_path, 'r') as file:
        for line in file:
            # Extracting the cur_tour value from each line
            cur_tour = int(line.split()[-1])
            cur_tour_values.append(cur_tour)
            
            # Calculate average for every twenty consecutive values
            if len(cur_tour_values) == 20:
                average = calculate_average(cur_tour_values)
                f.write(f"Average for experiment {experiments[idx]} is {average}\n")
                idx += 1
                cur_tour_values = []  # Reset the list for the next 20 values

    # If there are remaining values after the loop, calculate the average for them
    if cur_tour_values:
        average = calculate_average(cur_tour_values)
        print(f"Average for the last {len(cur_tour_values)} values: {average}")
    f.close()  

def genetic_algorithm_path():
    best_comb = (0,0,0)
    best_path = inf
    f = open ("results.txt", "w")
    population = [100,300,500]
    generations = [100,500,1000]
    mutation_rate = [0.05,0.09,0.1]
    for pop in population:
        for gen in generations:
            for rate in mutation_rate:
                graph = Graph(gen)
                for i in range(20):
                    f.write(f"SIMPLE EVOLUTION LENGTH: population: {pop} generations: {gen} mutation_rate: {rate}")
                    curr_tour = calc_genetic_algorithm_tour(tsp, pop, gen, rate, graph)
                    f.write(f" cur_tour: {curr_tour}\n")
                graph.plot_graph(pop, gen, rate)
    f.write(f"Best combination: {best_comb} with path length: {best_path}")
    f.close()
    calc()

def print_results_from_tsp_path(tsp_path):
    tsp = read_tsp_file(tsp_path)

    print(f"TSP Problem:              {tsp["NAME"]}")
    print(f"PATH:                     {tsp_path}")
    print(f"NAIVE TOUR LENGTH:        {calc_naive_tour(tsp)}")
    print(f"NEAREST NEIGHBOR LENGTH:  {calc_nearest_neighbor_tour(tsp)}")
    print(f"DP LENGTH :              {calc_held_karp_tour(tsp)}")
    print(f"CHRISTOFIDES LENGTH: {calc_christofides_algorithm_tour(tsp, 100, 10, Graph)}")  
    # write the results and the avrage results into output files: 'results.txt' and 'average_results.txt'
    genetic_algorithm_path()

def main():
    print_results_from_tsp_path("input.tsp")

if __name__ == "__main__":
    main()
