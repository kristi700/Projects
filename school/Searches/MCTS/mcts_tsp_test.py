import random
import math
import matplotlib.pyplot as plt

class Cities:
    def __init__(self, N, seed=42, x_range=(0, 20), y_range=(0, 20)):
        """
        Initialize the Cities class.

        Parameters:
        - N (int): Number of cities to generate.
        - seed (int): Seed for random number generator for reproducibility.
        - x_range (tuple): Range for x coordinates.
        - y_range (tuple): Range for y coordinates.
        """
        self.N = N
        self.seed = seed
        self.x_range = x_range
        self.y_range = y_range
        self.city_coords = {}
        self.distance_matrix = {}
        self.cities = []
        self.generate_cities()
        self.calculate_distance_matrix()

    def generate_cities(self):
        random.seed(self.seed)
        self.cities = [f"City_{i}" for i in range(self.N)]
        self.city_coords = {}
        for city in self.cities:
            x = random.uniform(*self.x_range)
            y = random.uniform(*self.y_range)
            self.city_coords[city] = (x, y)

    def euclidean_distance(self, coord1, coord2):
        return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

    def calculate_distance_matrix(self):
        self.distance_matrix = {}
        for i in range(len(self.cities)):
            for j in range(i, len(self.cities)):
                city1 = self.cities[i]
                city2 = self.cities[j]
                coord1 = self.city_coords[city1]
                coord2 = self.city_coords[city2]
                distance = self.euclidean_distance(coord1, coord2)
                self.distance_matrix[(city1, city2)] = distance
                self.distance_matrix[(city2, city1)] = distance

    def calculate_total_distance(self, tour):
        distance = 0
        for i in range(len(tour) - 1):
            city1 = tour[i]
            city2 = tour[i + 1]
            distance += self.distance_matrix[(city1, city2)]
        distance += self.distance_matrix[(tour[-1], tour[0])]
        return distance

    def generate_random_tour(self):
        tour = self.cities.copy()
        random.shuffle(tour)
        return tour

    def plot_tour(self, tour, filename='tour.png'):
        x = [self.city_coords[city][0] for city in tour]
        y = [self.city_coords[city][1] for city in tour]
        x.append(self.city_coords[tour[0]][0])
        y.append(self.city_coords[tour[0]][1])

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o-', color='blue', linewidth=2, markersize=8)
        for city in tour:
            plt.text(self.city_coords[city][0] + 0.1, self.city_coords[city][1] + 0.1,
                     city, fontsize=9)
        plt.title('Traveling Salesman Tour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

### MCTS

class Node():
    def __init__(self, state, unvisited_cities, parent=None):
        self.state = state # List of visited cities
        self.unvisited_cities = unvisited_cities
        self.visits = 0
        self.parent = parent
        self.children = []
        self.is_terminal = len(unvisited_cities) == 0
        self.total_reward = 0.0
        self.is_fully_expanded = False

# funcs
def select(node):
    """
    In case node is not a terminal state, either expands it (in case it is not fully expanded), or returns its best child.
    """
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return expand(node)
        else:
            node = best_child(node, C = math.sqrt(2)) # C being the constant in UCB1
    return node


def expand(node):
    """
    Node not being a terminal state is assumed (checked in selection)

    Adds new child to node
    """
    unvisited_cities = node.unvisited_cities
    for city in unvisited_cities.copy():
        new_state = node.state + [city]
        new_unvisited = unvisited_cities.copy()
        new_unvisited.remove(city)
        child_node = Node(state=new_state, unvisited_cities=new_unvisited, parent=node)
        node.children.append(child_node)
        if not new_unvisited:
            child_node.is_terminal = True
    node.is_fully_expanded = True
    return random.choice(node.children)
        
def best_child(node, C):
    """
    Selects best child based on UCB1 formula
    """
    choice_weights = []
    for child in node.children:
        exploitation = child.total_reward / (child.visits+1e-10)
        exploration = C * math.sqrt(2*math.log(node.visits)/(child.visits+1e-10))
        choice_weights.append(exploitation + exploration)
    max_weight = max(choice_weights)
    # Needed as more can have same weights
    best_children = [child for child, weight in zip(node.children, choice_weights) if weight == max_weight]
    return random.choice(best_children)

# NOTE - refactor, we dont need cities_instance here!!!
def simulate(node, cities_instance: Cities):
    """
    Simulation
    """
    current_state = node.state.copy()
    unvisited = node.unvisited_cities
    random.shuffle(unvisited)
    current_state.extend(unvisited)
    total_distance =  cities_instance.calculate_total_distance(current_state)
    reward = -total_distance # Since we wanna minize total_distance, reward is - (mcts maximizes)
    return reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent

def mcts(root, iterations, cities_instance):
    for _ in range(iterations):
        leaf = select(root)
        reward = simulate(leaf, cities_instance)
        backpropagate(leaf, reward)

def get_best_child(node):
    most_visited_children = max(node.children, key=lambda c: c.visits)
    return most_visited_children

def get_tour(node):
    tour = node.state.copy()
    while node.children:
        node = get_best_child(node)
        if node.state[-1] not in tour:
            tour.append(node.state[-1])
    return tour

def main():
    N = 10
    seed = 42

    cities_instance = Cities(N=N, seed=seed)
    random_tour = cities_instance.generate_random_tour()

    total_distance = cities_instance.calculate_total_distance(random_tour)

    print(f"Total Distance - Random: {total_distance:.2f}")

    cities_instance.plot_tour(random_tour, filename='random_tour.png')

    # MCTS
    starting_city = cities_instance.cities[0]
    initial_state = [starting_city]
    unvisited_cities = list(cities_instance.cities)
    unvisited_cities.remove(starting_city)

    root = Node(initial_state, unvisited_cities)

    iterations = 1000
    mcts(root, iterations, cities_instance)
    best_tour = get_tour(root)
    best_distance = cities_instance.calculate_total_distance(best_tour)

    print(f"Total Distance - MCTS: {best_distance:.2f}")

    cities_instance.plot_tour(best_tour, filename='mcts_best_tour.png')
if __name__ == "__main__":
    main()