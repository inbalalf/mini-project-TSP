import matplotlib.pyplot as plt

from city import CityCoords

def plot_tour(tsp_dict: dict, tour_indices: list[int]):
    """
    Plot the tour given by the indices in tour_indices
    Args:
        tsp_dict (dict): A dictionary containing the parsed TSP file
        tour_indices (list[int]): A list of indices for the cities in the tour
    
    """
    # Extract coordinates for the tour
    tour_coordinates: list[CityCoords] = [tsp_dict["CITIES"][index-1] for index in tour_indices]

    # Add the starting city at the end to complete the tour
    tour_coordinates.append(tour_coordinates[0])

    # Unzip the coordinates into separate lists for x and y
    x, y = zip(*[(city.x, city.y) for city in tour_coordinates])
    # Plot the tour
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title('Tour between Cities')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()


