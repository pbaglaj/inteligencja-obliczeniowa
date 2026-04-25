import matplotlib.pyplot as plt
import random
import math
from aco import AntColony

plt.style.use("dark_background")

# 1. Losowanie 15 miast z przedziału [0, 100]
random.seed(42) # Ustawiamy ziarno, aby każda konfiguracja testowała tę samą mapę
NUM_NODES = 15
COORDS = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(NUM_NODES)]

# Funkcja pomocnicza obliczająca długość trasy (aby móc ocenić skuteczność)
def calculate_path_distance(path):
    dist = 0.0
    for i in range(len(path) - 1):
        dist += math.hypot(path[i][0] - path[i+1][0], path[i][1] - path[i+1][1])
    # Dodajemy powrót do pierwszego miasta
    dist += math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    return dist

# 2. Definicja 4 różnych zestawów parametrów do eksperymentu
experiments = [
    {"name": "1. Baza (Zbalansowane)", "alpha": 1.0, "beta": 1.0, "evap": 0.40},
    {"name": "2. Siła stada (Alfa=3.0)", "alpha": 3.0, "beta": 1.0, "evap": 0.40},
    {"name": "3. Siła odległości (Beta=3.0)", "alpha": 1.0, "beta": 3.0, "evap": 0.40},
    {"name": "4. Szybkie zapominanie (Evap=0.8)", "alpha": 1.0, "beta": 1.0, "evap": 0.80}
]

# Przygotowanie siatki wykresów 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

print(f"Rozpoczynam eksperymenty dla {NUM_NODES} miast...\n")

for idx, exp in enumerate(experiments):
    # Inicjalizacja kolonii z nowymi parametrami
    colony = AntColony(
        COORDS, 
        ant_count=200, 
        alpha=exp["alpha"], 
        beta=exp["beta"], 
        pheromone_evaporation_rate=exp["evap"], 
        pheromone_constant=1000.0,
        iterations=150
    )
    
    optimal_nodes = colony.get_path()
    distance = calculate_path_distance(optimal_nodes)
    
    print(f"{exp['name']:<35} -> Odległość: {distance:.2f}")
    
    # Rysowanie wykresu dla danego eksperymentu
    ax = axes[idx]
    ax.set_title(f"{exp['name']}\nDługość trasy: {distance:.2f}", color="white", pad=10)
    
    # Rysowanie punktów
    for x, y in COORDS:
        ax.plot(x, y, "g.", markersize=12)
        
    # Rysowanie ścieżek
    for i in range(len(optimal_nodes) - 1):
        ax.plot((optimal_nodes[i][0], optimal_nodes[i+1][0]), 
                (optimal_nodes[i][1], optimal_nodes[i+1][1]), color="cyan", alpha=0.8)
    # Rysowanie powrotu do bazy
    ax.plot((optimal_nodes[-1][0], optimal_nodes[0][0]), 
            (optimal_nodes[-1][1], optimal_nodes[0][1]), color="cyan", alpha=0.8)
    
    ax.axis("off")

plt.tight_layout()
plt.show()