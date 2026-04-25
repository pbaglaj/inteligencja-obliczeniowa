import pygad
import math

# Funkcja obliczająca wytrzymałość (zgodnie ze wzorem inżynierów)
def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)

# Funkcja fitness dla algorytmu genetycznego
def fitness_func(ga_instance, solution, solution_idx):
    # 'solution' to tablica 6 liczb rzeczywistych oznaczających udziały metali
    # Rozpakowujemy listę za pomocą operatora * do argumentów funkcji
    return endurance(*solution)

# --- KONFIGURACJA ALGORYTMU GENETYCZNEGO ---

# Definicja przedziału [0, 1) dla wszystkich genów
gene_space = {'low': 0.0, 'high': 0.99999}

num_genes = 6             # 6 zmiennych (x, y, z, u, v, w)
sol_per_pop = 50          # Wielkość populacji (50 osobników/stopów)
num_parents_mating = 25   # Połowa populacji zostaje rodzicami
num_generations = 150     # Liczba pokoleń
keep_parents = 2          # Elityzm - zachowujemy 2 najlepszych rodziców bez zmian

# Ustawienie mutacji powyżej 16.67%, aby przy 6 genach zmutował minimum 1 gen
mutation_percent_genes = 20

# Inicjalizacja instancji PyGAD
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    mutation_percent_genes=mutation_percent_genes,
    # Opcjonalnie: możemy użyć "random" lub "adaptive"
    mutation_type="random", 
    suppress_warnings=True  # Ukrywa drobne ostrzeżenia z biblioteki
)

# Uruchomienie ewolucji
ga_instance.run()

# --- PODSUMOWANIE WYNIKÓW ---

# Pobranie najlepszego znalezionego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("=== WYNIKI OPTYMALIZACJI ===")
print(f"Najlepsza znaleziona wytrzymałość stopu (Fitness): {solution_fitness}")
print("Najlepsze proporcje metali [x, y, z, u, v, w]:")
print(solution)

# Opcjonalne: Wyświetlenie wykresu, jak algorytm uczył się z pokolenia na pokolenie
ga_instance.plot_fitness(title="Wzrost wytrzymałości stopu w kolejnych pokoleniach")