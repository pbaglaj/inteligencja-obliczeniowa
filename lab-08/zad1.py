import pyswarms as ps
import numpy as np
import math
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

# 1. Poprawiona funkcja endurance - pobiera jedną tablicę 6 argumentów
def endurance(params):
    x, y, z, u, v, w = params
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

# 2. Funkcja wrapper (f), która przebiega pętlą po całym roju
def f(swarm):
    # Pobieramy liczbę cząstek w roju
    n_particles = swarm.shape[0]
    
    # Tworzymy pustą tablicę na wyniki
    results = np.zeros(n_particles)
    
    # Przebiegamy pętlą po wszystkich cząstkach
    for i in range(n_particles):
        # Pobieramy parametry dla pojedynczej cząstki
        particle_params = swarm[i]
        
        # Wywołujemy funkcję i wstawiamy wynik do tablicy.
        # DODAJEMY ZNAK MINUS (-), ponieważ chcemy ZMAKSYMALIZOWAĆ wytrzymałość!
        results[i] = -endurance(particle_params)
        
    return results

# Ustawienia ograniczeń (min 0, max 1)
x_min = np.zeros(6)
x_max = np.ones(6)
my_bounds = (x_min, x_max)

# Ustawienia algorytmu
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Inicjalizacja optymalizatora (dimensions=6)
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

# 3. Wrzuć f do optymalizatora
cost, pos = optimizer.optimize(f, iters=1000)

print("\n--- WYNIKI ---")
# Odwracamy znak 'cost', aby wyświetlić rzeczywistą wytrzymałość na plusie
print(f"Maksymalna znaleziona wytrzymałość stopu: {-cost}")
print(f"Optymalne proporcje metali [x, y, z, u, v, w]:\n{pos}")

# 1. Wyrysowanie historii kosztu za pomocą wbudowanej funkcji pyswarms
plot_cost_history(cost_history=optimizer.cost_history)

# 2. Dodanie tytułów dla czytelności
plt.title("Historia optymalizacji PSO (Minimalizacja kosztu)")
plt.xlabel("Iteracje")
plt.ylabel("Koszt (-Endurance)")

# 3. Wyświetlenie wykresu
plt.show()