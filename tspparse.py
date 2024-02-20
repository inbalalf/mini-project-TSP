from collections import deque
from city        import CityCoords

def build_dict():
    return { "COMMENT"          : ""
           , "DIMENSION"        : None
           , "EDGE_WEIGHT_TYPE" : None
           , "CITIES"           : []}

def parse_tsp_file(tsp,tspfile):
    for line in tspfile:
        words   = deque(line.split())
        keyword = words.popleft().strip(": ")

        if keyword == "COMMENT":
            tsp["COMMENT"] += " ".join(words).strip(": ")
        elif keyword == "NAME":
            tsp["NAME"] = " ".join(words).strip(": ")
        elif keyword == "DIMENSION":
            tsp["DIMENSION"] = int(" ".join(words).strip(": "))
        elif keyword == "EDGE_WEIGHT_TYPE":
            tsp["EDGE_WEIGHT_TYPE"] = " ".join(words).strip(": ")
        elif keyword == "NODE_COORD_SECTION":
            break

def read_int(words):
    return int(words.popleft())

def read_city_coords(city_index, words):
    city_number = read_int(words)
    if city_number == city_index:
        x = float(words.popleft())
        y = float(words.popleft())
        return CityCoords(x, y)
    else:
        print(f"Missing or mislabeld city: expected {city_index}")

def read_cities(dict, file):
    for i in range(1, dict["DIMENSION"] + 1):
        line  = file.readline()
        words = deque(line.split())
        if dict["EDGE_WEIGHT_TYPE"] == "EUC_2D":
            dict["CITIES"].append(read_city_coords(i, words))
        else:
            print("Unsupported coordinate type: " + dict["EDGE_WEIGHT_TYPE"])
            
def read_tsp_file(path):
    tsp_file = open(path,'r')
    tsp_dict = build_dict()
    parse_tsp_file(tsp_dict,tsp_file)
    read_cities(tsp_dict,tsp_file)
    tsp_file.close()
    return tsp_dict
