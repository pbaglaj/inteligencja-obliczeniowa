import numpy as np
import random

maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,1,0,0,1],
    [1,1,1,0,0,0,1,0,1,1,0,1],
    [1,0,0,0,1,0,1,0,0,0,0,1],
    [1,0,1,0,1,1,0,0,1,1,0,1],
    [1,0,0,1,1,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,1,1,0,0,0,1,1,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1], # [10][10] to wyjście (zero obok ramki)
    [1,1,1,1,1,1,1,1,1,1,1,1]
])

START = (1, 1)
EXIT = (10, 10)
MAX_STEPS = 30
NUM_ANTS = 50
ITERATIONS = 100

# Parametry ACO
ALPHA = 1.0  # Waga feromonu
BETA = 2.0   # Waga heurystyki (ciąg do wyjścia)
EVAPORATION = 0.1
PHEROMONE_INIT = 0.1

# Inicjalizacja feromonów dla wszystkich dozwolonych ruchów
pheromones = {}

def get_neighbors(pos):
    """Zwraca listę dozwolonych kroków z danej pozycji."""
    r, c = pos
    neighbors = []
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]: # Góra, Dół, Lewo, Prawo
        nr, nc = r + dr, c + dc
        if maze[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def distance_to_exit(pos):
    """Odległość Manhattan do wyjścia (heurystyka)."""
    return abs(pos[0] - EXIT[0]) + abs(pos[1] - EXIT[1])

# --- GŁÓWNA PĘTLA ACO ---
best_path = None
best_length = float('inf')

for iteration in range(ITERATIONS):
    all_paths = []
    
    for ant in range(NUM_ANTS):
        current_pos = START
        path = [current_pos]
        
        # Mrówka idzie dopóki nie znajdzie wyjścia lub nie przekroczy max kroków
        for step in range(MAX_STEPS):
            if current_pos == EXIT:
                break
                
            neighbors = get_neighbors(current_pos)
            # Zabezpieczenie przed cofaniem się (żeby mrówka nie chodziła w kółko)
            unvisited_neighbors = [n for n in neighbors if n not in path[:-1] or n != path[-2]]
            
            if not unvisited_neighbors:
                break # Ślepa uliczka
                
            # Obliczanie prawdopodobieństw przejścia
            probabilities = []
            for n in unvisited_neighbors:
                edge = (current_pos, n)
                phero = pheromones.get(edge, PHEROMONE_INIT)
                
                dist = distance_to_exit(n)
                # Heurystyka: im bliżej wyjścia tym większa wartość
                heur = 1.0 / (dist + 0.1) 
                
                prob = (phero ** ALPHA) * (heur ** BETA)
                probabilities.append(prob)
                
            # Normalizacja i wybór kolejnego kroku (Ruletka)
            total_prob = sum(probabilities)
            probabilities = [p / total_prob for p in probabilities]
            next_pos = random.choices(unvisited_neighbors, weights=probabilities)[0]
            
            path.append(next_pos)
            current_pos = next_pos
            
        # Jeśli mrówka dotarła do wyjścia, zapisujemy jej trasę
        if path[-1] == EXIT:
            all_paths.append(path)
            if len(path) < best_length:
                best_length = len(path)
                best_path = path

    # Parowanie feromonów
    for edge in pheromones:
        pheromones[edge] *= (1.0 - EVAPORATION)
        
    # Wzmocnienie feromonów na ścieżkach, które dotarły do celu
    for path in all_paths:
        # Mrówki, które doszły szybciej, zostawiają więcej feromonu
        phero_deposit = 10.0 / len(path) 
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            pheromones[edge] = pheromones.get(edge, PHEROMONE_INIT) + phero_deposit

print(f"Najkrótsza znaleziona ścieżka: {best_length - 1} kroków.")
print(f"Trasa: {best_path}")