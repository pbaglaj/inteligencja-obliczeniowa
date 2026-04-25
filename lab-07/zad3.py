import pygad
import numpy as np
import time

# a) Kodowanie labiryntu jako macierz 12x12
# 0 = wolne pole, 1 = ściana (ramka wokół to same jedynki)
maze = [
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
]

# e) Funkcja Fitness
def fitness_func(ga_instance, solution, solution_idx):
    # Agent startuje w lewym górnym rogu wewnętrznej przestrzeni
    row, col = 1, 1 
    
    for move in solution:
        new_row, new_col = row, col
        
        # 0: Góra, 1: Prawo, 2: Dół, 3: Lewo
        if move == 0:
            new_row -= 1
        elif move == 1:
            new_col += 1
        elif move == 2:
            new_row += 1
        elif move == 3:
            new_col -= 1
            
        # Zabezpieczenie przed wyjściem za listę (właściwie zapewnia to ramka z jedynek)
        # Sprawdzamy czy nie wchodzimy w ścianę
        if maze[new_row][new_col] == 0:
            row, col = new_row, new_col  # Aktualizacja pozycji, ruch jest legalny
            
        # Sprawdzamy czy dotarliśmy do mety (współrzędne 10, 10)
        if row == 10 and col == 10:
            return 100  # Maksymalna, wysoce premiowana wartość za znalezienie wyjścia
            
    # Jeśli algorytm po 30 krokach nie dotarł do wyjścia:
    # Obliczamy karę na podstawie odległości "Manhattan" od wyjścia. 
    # Wartość ujemna, aby faworyzować rozwiązania, które lądują bliżej mety.
    distance = abs(10 - row) + abs(10 - col)
    return -distance


# b, c, d) Ustawienia Algorytmu Genetycznego
gene_space = [0, 1, 2, 3] # Nasze 4 kierunki
num_genes = 30 # Szukamy drogi o maksymalnie 30 krokach

sol_per_pop = 150       # Zwiększona populacja
num_parents_mating = 100 # Ok. 30% populacji do rozmnażania
num_generations = 20    # Maksymalna liczba pokoleń, bylo 200
keep_parents = 2        # Elityzm

mutation_percent_genes = 5 # 5% z 30 daje ok. 1.5 gena

# Opcjonalne mapowanie ruchów na słowa dla lepszej czytelności wyniku
directions = {0: "Góra", 1: "Prawo", 2: "Dół", 3: "Lewo"}


# f) Benchmark - pętla 10 prób z mierzeniem czasu
print("Rozpoczynam testowanie algorytmu (10 prób)...\n")
success_times = []

for i in range(1, 11):
    # Nowa instancja na każdy obrót, by zresetować stan
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        keep_parents=keep_parents,
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=["reach_100"], # f) Warunek stopu po znalezieniu mety
        suppress_warnings=True
    )
    
    start = time.time()
    ga_instance.run()
    end = time.time()
    
    solution, solution_fitness, _ = ga_instance.best_solution()
    
    if solution_fitness == 100:
        elapsed = end - start
        success_times.append(elapsed)
        print(f"Próba {i}: ZNALEZIONO WYJŚCIE! Czas: {elapsed:.4f} s")
        if i == 1:
            path = [directions[move] for move in solution]
            print(f"Przykładowa zwycięska ścieżka: {path}\n")
    else:
        print(f"Próba {i}: Algorytm utknął z najlepszym fitness: {solution_fitness}")

ga_instance.plot_fitness()

print("-" * 40)
if len(success_times) > 0:
    avg_time = sum(success_times) / len(success_times)
    print(f"Skuteczność: {len(success_times)}/10")
    print(f"Średni czas udanego działania: {avg_time:.4f} sekundy")
else:
    print("Skuteczność: 0/10. Algorytm nie poradził sobie ze znalezieniem drogi.")