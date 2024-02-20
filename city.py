from math  import pi, sqrt, cos, sin, acos
from numpy import around

class CityCoords:
    def __init__(self, x = None, y = None):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return CityCoords(self.x - other.x, self.y - other.y)

    def norm(self):
        return sqrt(self.x ** 2 + self.y ** 2)

class Distance:

    def euc_2d_distance(city1, city2):
        return int(around((city1 - city2).norm()))

    def distance(city1, city2):
        if type(city1) != type(city2):
            print("Can't calculate distance between cities: different coord types")
        elif type(city1) == CityCoords:
            return Distance.euc_2d_distance(city1, city2)
        
    def calc_distance(city1: CityCoords, city2: CityCoords):
        """Calculate distance between cities by their (one-based) indices"""
        return Distance.distance(city1, city2)


