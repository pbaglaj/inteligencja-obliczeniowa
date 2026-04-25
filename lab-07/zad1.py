import pygad
import numpy
import time

# --- DANE WEJŚCIOWE ---
items = ["zegar", "obraz-pejzaż", "obraz-portret", "radio", "laptop", "lampka nocna", 
         "srebrne sztućce", "porcelana", "figura z brązu", "skórzana torebka", "odkurzacz"]

# Wartości i wagi jako tablice numpy dla ułatwienia obliczeń wektorowych
values = numpy.array([100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300])
weights = numpy.array([7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15])
weight_limit = 25

# --- DEFINICJA FUNKCJI FITNESS ---
def fitness_func(ga_instance, solution, solution_idx):
    total_weight = numpy.sum(solution * weights)
    total_value = numpy.sum(solution * solution * values)
    
    # Kary za przekroczenie wagi
    if total_weight > weight_limit:
        return 0
    else:
        return total_value

# --- PARAMETRY ALGORYTMU ---
gene_space = [0, 1]
sol_per_pop = 30           # wielkość populacji
num_genes = len(items)     # 11 genów
num_parents_mating = 15    # około 50% populacji
num_generations = 50       # liczba pokoleń
keep_parents = 2           # elityzm
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10 # ~1 gen mutuje

# =====================================================================
# CZĘŚĆ 1: Pojedyncze uruchomienie (odpowiedź na punkty a, b, c, d)
# =====================================================================
print("--- POJEDYNCZE URUCHOMIENIE ---")
ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Najlepsze znalezione rozwiązanie (chromosom): {solution}")
print(f"Wartość najlepszego rozwiązania: {solution_fitness} zł")

# Analiza spakowanych przedmiotów
packed_items = []
total_weight = 0
for idx, is_taken in enumerate(solution):
    if is_taken == 1:
        packed_items.append(items[idx])
        total_weight += weights[idx]

print(f"Spakowane przedmioty: {packed_items}")
print(f"Łączna waga: {total_weight} kg")

# Wyświetlenie wykresu
ga_instance.plot_fitness()

# =====================================================================
# CZĘŚĆ 2: Benchmark - 10 prób i mierzenie czasu (odpowiedź na e, f)
# =====================================================================
print("\n--- BENCHMARK (10 prób) ---")

optimal_target = 1630
success_count = 0
times_of_success = []

for i in range(10):
    # Za każdym razem tworzymy nową instancję, aby zresetować stan algorytmu
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=100, # dajemy mu więcej czasu na znalezienie celu
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=[f"reach_{optimal_target}"] # Zatrzymanie gdy znajdzie 1630
    )
    
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    
    best_sol, best_fit, _ = ga_instance.best_solution()
    
    if best_fit >= optimal_target:
        success_count += 1
        times_of_success.append(end_time - start_time)

# Wyniki Benchmarku
success_percentage = (success_count / 10) * 100
print(f"Skuteczność algorytmu (znalezienie minimum {optimal_target} zł): {success_percentage}%")

if success_count > 0:
    avg_time = sum(times_of_success) / success_count
    print(f"Średni czas działania przy udanej próbie: {avg_time:.5f} sekund")
else:
    print("Nie udało się znaleźć optymalnego rozwiązania w żadnej z prób.")