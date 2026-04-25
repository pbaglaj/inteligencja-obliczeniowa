import matplotlib.pyplot as plt
import math
from aco import AntColony

plt.style.use("dark_background")

# Dokładny grid 5x5
COORDS = (
    (0, 0), (10, 0), (20, 0), (30, 0), (40, 0),
    (0, 10), (10, 10), (20, 10), (30, 10), (40, 10),
    (0, 20), (10, 20), (20, 20), (30, 20), (40, 20),
    (0, 30), (10, 30), (20, 30), (30, 30), (40, 30),
    (0, 40), (10, 40), (20, 40), (30, 40), (40, 40),
)

def calculate_path_distance(path):
    dist = 0.0
    for i in range(len(path) - 1):
        dist += math.hypot(path[i][0] - path[i+1][0], path[i][1] - path[i+1][1])
    dist += math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    return dist

def plot_nodes():
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")

plot_nodes()

# Konfiguracja dla trudnego problemu (więcej mrówek i iteracji)
colony = AntColony(
    COORDS, 
    ant_count=400,        # Dużo mrówek do eksploracji
    alpha=1.0,            # Feromony
    beta=2.0,             # Ważniejsza odległość (chcemy unikać długich skosów)
    pheromone_evaporation_rate=0.5, 
    pheromone_constant=1000.0,
    iterations=300        # Sporo czasu na szukanie
)

print("Szukam optymalnej trasy na siatce 5x5...")
optimal_nodes = colony.get_path()
dist = calculate_path_distance(optimal_nodes)

print(f"Znaleziona długość trasy: {dist:.2f}")
print(f"Idealne minimum matematyczne: 254.14")
print(f"Klasyczna trasa 'ludzka': 280.00")

# Rysowanie trasy
for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
        color="cyan", linewidth=2
    )

# Zamknięcie cyklu
plt.plot(
    (optimal_nodes[-1][0], optimal_nodes[0][0]),
    (optimal_nodes[-1][1], optimal_nodes[0][1]),
    color="cyan", linewidth=2
)

plt.title(f"ACO na Gridzie 5x5\nDługość znalezionej trasy: {dist:.2f}", color="white")
plt.show()