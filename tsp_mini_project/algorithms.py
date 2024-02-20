import itertools
import random

import numpy as np
from city import CityCoords, Distance
from generation import Generation
from graph import Graph
from plot_tour import plot_tour
from city import Distance
import networkx as nx
from itertools import combinations


#####################
# general functions #
#####################
def calc_path_length(cities, path):
    """
    Find the length of a path of cities given as a list
    Parameters:
    - cities (list): A list of city objects.
    - path (list): A list of indices representing the order in which cities are visited.

    Returns:
    - total_distance (float): The total length (distance) of the given path.
    """
    total_distance = 0
    num_cities = len(cities)
    for i in range(0, num_cities - 1):
        total_distance += Distance.distance(
            cities[path[i] - 1], cities[path[i + 1] - 1]
        )
    total_distance += Distance.distance(
        cities[path[-1] - 1], cities[path[0] - 1]
    )  # Return to the starting city
    return total_distance


def create_tour(path):
    """Append the first city in a path to the end in order to obtain a tour"""
    path.append(path[0])
    return path


#####################
# naive algorithms  #
#####################


def naive_tour(tsp_dict: dict):
    dim = tsp_dict["DIMENSION"]
    all_cities_indices = list(range(1, dim + 1))
    all_permutations = list(itertools.permutations(all_cities_indices))
    cities = tsp_dict["CITIES"]
    min_distance = float("inf")
    best_tour = None

    for tour in all_permutations:
        current_distance = calc_path_length(cities, tour)
        if current_distance < min_distance:
            min_distance = current_distance
            best_tour = tour

    return best_tour


def calc_naive_tour(tsp_dict):
    path = naive_tour(tsp_dict)
    plot_tour(tsp_dict, path)
    return calc_path_length(tsp_dict["CITIES"], path)


#####################
# DP algorithms  #
#####################


def held_karp_path(tsp_dict: dict) -> list[int]:
    """
    Finds an optimal tour using the Held-Karp dynamic programming algorithm for solving the Traveling Salesman Problem (TSP).

    Parameters:
    - tsp_dict (dict): A dictionary containing information about the TSP, including a list of city coordinates under the key "CITIES".

    Returns:
    - optimal_path (list): The optimal tour as a list of city indices.
    """
    cities: list[CityCoords] = tsp_dict["CITIES"]
    n = len(cities)

    # Initialize memoization table
    memo = {}
    min_distance, optimal_path = held_karp_helper(1, 1, memo, cities, n)
    # Start the recursion from the first city
    return optimal_path


def held_karp_helper(mask, current_city, memo, cities, n) -> (float, list[int]):
    """
    Helper function for the Held-Karp algorithm to recursively calculate the minimum distance and optimal path.

    Parameters:
    - mask (int): Bitmask representing visited cities.
    - current_city (int): Index of the current city.
    - memo (dict): Memoization table to store already computed results.
    - cities (list): List of city coordinates.
    - n (int): Total number of cities.

    Returns:
    - min_distance (float): Minimum distance for the current state.
    - optimal_path (list): Optimal path for the current state.
    """
    # If all cities have been visited
    if mask == (1 << n) - 1:
        return Distance.calc_distance(cities[current_city - 1], cities[0]), [
            current_city,
            1,
        ]

    # If the result is already memoized
    if (mask, current_city) in memo:
        return memo[(mask, current_city)]

    min_distance = float("inf")
    optimal_path = []

    for next_city in range(1, n + 1):  # Adjust the range to start from 1
        if (mask >> (next_city - 1)) & 1 == 0:  # If the next city is not visited
            new_mask = mask | (1 << (next_city - 1))
            dist, path = held_karp_helper(new_mask, next_city, memo, cities, n)
            dist += Distance.calc_distance(
                cities[current_city - 1], cities[next_city - 1]
            )

            if dist < min_distance:
                min_distance = dist
                optimal_path = [current_city] + path

    # Memoize the result
    memo[(mask, current_city)] = min_distance, optimal_path

    return min_distance, optimal_path


def calc_held_karp_tour(tsp_dict):
    path = held_karp_path(tsp_dict)
    plot_tour(tsp_dict, path)
    return calc_path_length(tsp_dict["CITIES"], path)


#####################
# greedy algorithms #
#####################


def find_nearest_neighbor(
    cities: list[CityCoords], untraveled_cities: set[int], current_city: int
):
    """
    Given a set of city keys, find the key corresponding to the closest city.

    Parameters:
    - cities (list): A list of city coordinates.
    - untraveled_cities (set): A set of city keys that haven't been visited.
    - current_city (int): The key of the current city.

    Returns:
    - nearest_city (int): The key corresponding to the nearest city from the current city.
    """

    distance_to_current_city = lambda city: Distance.calc_distance(
        cities[current_city - 1], cities[city - 1]
    )
    return min(untraveled_cities, key=distance_to_current_city)


def nearest_neighbor_tour(tsp_dict: dict):
    """
    Construct a tour through all cities in a TSP by following the nearest neighbor heuristic.

    Parameters:
    - tsp_dict (dict): A dictionary containing information about the TSP, including the number of cities under the key "DIMENSION".

    Returns:
    - tour (list): A tour through all cities constructed using the nearest neighbor heuristic.
    """
    nearest_neighbor_path = [1]
    current_city: int = 1
    cities_to_travel: set[int] = set(range(2, tsp_dict["DIMENSION"] + 1))
    cities: list[CityCoords] = tsp_dict["CITIES"]

    while cities_to_travel:
        current_city = find_nearest_neighbor(cities, cities_to_travel, current_city)
        nearest_neighbor_path.append(current_city)
        cities_to_travel.remove(current_city)

    return create_tour(nearest_neighbor_path)


def calc_nearest_neighbor_tour(tsp_dict):
    path = nearest_neighbor_tour(tsp_dict)
    plot_tour(tsp_dict, path)
    return calc_path_length(tsp_dict["CITIES"], path)


#####################
# approximation algorithm #
#####################


def build_graph(cities):
    graph = nx.Graph()
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                graph.add_edge(
                    city1, city2, weight=Distance.calc_distance(city1, city2)
                )
    return graph


def minimum_spanning_tree(graph):
    return nx.minimum_spanning_tree(graph)


def minimum_weight_perfect_matching(odd_vertices):
    min_weight_matching = nx.Graph()
    odd_pairs = list(combinations(odd_vertices, 2))

    for pair in odd_pairs:
        min_weight_matching.add_edge(
            pair[0], pair[1], weight=Distance.calc_distance(pair[0], pair[1])
        )

    min_weight_matching = nx.min_weight_matching(min_weight_matching)

    return min_weight_matching


def eulerian_circuit(graph) -> list[CityCoords]:
    circuit_edges = list(nx.eulerian_circuit(graph))
    circuit = [circuit_edges[0][0]] + [edge[1] for edge in circuit_edges]
    return circuit


def shortcutting(eulerian_circuit, num_vertices):
    shortcut_circuit = []
    visited = set()

    for vertex in eulerian_circuit:
        if vertex not in visited:
            visited.add(vertex)
            shortcut_circuit.append(vertex)

    return shortcut_circuit


def christofides_algorithm(cities):
    # Create a complete graph with cities as nodes
    graph = build_graph(cities)

    num_cities = len(cities)

    # Step 1: Minimum Spanning Tree
    mst = minimum_spanning_tree(graph)

    # Step 2: Minimum Weight Perfect Matching
    odd_vertices = [
        node for node, degree in dict(mst.degree()).items() if degree % 2 != 0
    ]
    min_weight_matching = minimum_weight_perfect_matching(odd_vertices)

    # Convert the set of edges to a list
    min_weight_matching_edges = list(min_weight_matching)

    # Combine the MST and minimum weight matching
    augmented_graph = nx.Graph(mst.edges)
    augmented_graph.add_edges_from(min_weight_matching_edges)

    # Step 3: Eulerian Circuit
    eulerian_circuit_path: list[CityCoords] = eulerian_circuit(augmented_graph)

    # Step 4: Shortcutting
    shortcut_circuit = shortcutting(eulerian_circuit_path, num_cities)

    # Convert city names to integers
    city_dict = {city: i for i, city in enumerate(cities)}
    shortcut_circuit_int = [city_dict[city] for city in shortcut_circuit]

    return shortcut_circuit_int


def calc_christofides_algorithm_tour(tsp_dict):
    path = christofides_algorithm(tsp_dict["CITIES"])
    plot_tour(tsp_dict, path)
    return calc_path_length(tsp_dict["CITIES"], path)


#####################
# Evolutionary algorithms  #
#####################


def initialize_population(population_size, num_cities) -> list[list[int]]:
    """
    Initializes the population for a genetic algorithm solving the Traveling Salesman Problem (TSP).
    Randomly selects a permutation of cities- overall population_size permutations.

    Parameters:
    - population_size (int): The number of individuals (routes) in the population.
    - num_cities (int): The number of cities in the TSP.

    Returns:
    - population (list): A list of individuals, where each individual represents a route through all cities.
    """
    population = [
        random.sample(range(1, num_cities + 1), num_cities)
        for _ in range(population_size)
    ]
    return population


def selection(population, fitnesses):  # tournament selection
    tournament = random.sample(range(0, len(population)), k=len(population) // 2)
    tournament_fitnesses = [fitnesses[i] for i in tournament]
    winner_index = tournament[np.argmax(tournament_fitnesses)]
    return population[winner_index]


def crossover(parent1, parent2) -> (list[int], list[int]):
    """
    Perform crossover (recombination) between two parent routes to generate a child route.
    crossover Operation: Select a random crossover point and copy the first part of the first parent to the child.
        The seccound part will hold the cities from parent 2 that the child dont already have. this way the created child is a permutation of cities.
    Parameters:
    - parent1 (list): The first parent route for crossover.
    - parent2 (list): The second parent route for crossover.

    Returns:
    - child (list): The resulting child route after crossover.
    """
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point]
    child2 = parent2[:crossover_point]
    for city in parent2:
        if city not in child1:
            child1.append(city)
    for city in parent1:
        if city not in child2:
            child2.append(city)
    return child1, child2


def mutation(route: list[int], mutation_prob: float) -> list[int]:
    """
    Apply mutation to a given route with a certain probability.
    Mutation Operation: Swap the random positions of two cities in the given route.

    Parameters:
    - route (list): A list representing an individual route through cities.
    - mutation_prob (float): The probability of mutation for each individual.

    Returns:
    - route (list): The mutated route, possibly with the positions of two cities swapped.
    """
    if random.random() < mutation_prob:
        mutation_point1 = random.randint(0, len(route) - 1)
        mutation_point2 = random.randint(0, len(route) - 1)
        route[mutation_point1], route[mutation_point2] = (
            route[mutation_point2],
            route[mutation_point1],
        )
    return route


def genetic_algorithm_path(
    tsp_dict: dict,
    population_size: int,
    generations: int,
    mutation_rate: float,
    graph: Graph,
) -> list[int]:
    cities = tsp_dict["CITIES"]
    population: list[list[int]] = initialize_population(population_size, len(cities))
    for generation in range(generations):
        # Evaluate the fitness of each individual (path) in the population
        fitnesses: list[float] = []
        for i in range(population_size):
            fitnesses.append(1 / calc_path_length(cities, population[i]))
        gen = Generation(generation, fitnesses)
        graph.all_best_fitness[generation] += gen.best_fitness / 20
        graph.all_worst_fitness[generation] += gen.worst_fitness / 20
        graph.all_average_fitness[generation] += gen.average_fitness / 20

        # Select parents for crossover. Higher fitness values correspond to higher probabilities of parents selection.
        # parents: list[list[int]] = random.choices(population, weights=fitness, k=population_size)
        nextgen_population = []
        for i in range(int(population_size / 2)):
            parent1 = selection(population, fitnesses)  # select first parent
            parent2 = selection(population, fitnesses)  # select second parent
            offspring1, offspring2 = crossover(
                parent1, parent2
            )  # perform crossover between both parents
            nextgen_population += [
                mutation(offspring1, mutation_rate),
                mutation(offspring2, mutation_rate),
            ]  # mutate offspring
        population = nextgen_population

    # Find the best (minimal) route in the final population
    best_route = min(population, key=lambda route: calc_path_length(cities, route))

    return best_route


def calc_genetic_algorithm_tour(
    tsp_dict, population_size, generations, mutation_rate, graph
):
    path: list[int] = genetic_algorithm_path(
        tsp_dict, population_size, generations, mutation_rate, graph
    )
    plot_tour(tsp_dict, path)
    return calc_path_length(tsp_dict["CITIES"], path)
