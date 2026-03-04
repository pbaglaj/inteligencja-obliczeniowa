import math
import random
import matplotlib.pyplot as plt

odleglosc_celu = random.randint(50, 340)
margines_lewy = odleglosc_celu - 5
margines_prawy = odleglosc_celu + 5 
print(f"Cel namierzony! Odległość: {odleglosc_celu} metrów.")

h = 100
v0 = 50
g = 9.81

def oblicz_zasieg(v0, h, alfa_stopnie):
    alfa = math.radians(alfa_stopnie)
    zasieg = ((v0 * math.sin(alfa)) + (math.sqrt((v0**2) * (math.sin(alfa)**2) + 2 * h * g))) * ((v0 * math.cos(alfa)) / g)
    return max(0.0, zasieg)

def narysuj_trajektorie(v0, h, alfa_stopnie, zasieg_koncowy):
    alfa = math.radians(alfa_stopnie)
    
    if math.cos(alfa) == 0:
        t_total = (v0 * math.sin(alfa) + math.sqrt((v0 * math.sin(alfa))**2 + 2 * h * g)) / g
    else:
        t_total = zasieg_koncowy / (v0 * math.cos(alfa))
        
    czasy = [t_total * (i / 100) for i in range(101)]
    x = [v0 * math.cos(alfa) * t for t in czasy]
    y = [h + v0 * math.sin(alfa) * t - 0.5 * g * (t**2) for t in czasy]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue', label='Trajektoria pocisku')
    
    plt.grid(True)
    plt.title("Trajektoria strzału z maszyny Warwolf")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Wysokość [m]")
    
    plt.axvspan(margines_lewy, margines_prawy, color='green', alpha=0.2, label='Strefa celu')
    plt.legend()
    
    plt.savefig('trajektoria.png')
    print("Zapisano wykres do pliku 'trajektoria.png'.")
    plt.show()

proby = 0

while True:
    try:
        alfa_input = input("Podaj kąt alfa w stopniach (lub wpisz 'q' aby wyjść): ")
        if alfa_input.lower() == 'q':
            print("Poddano grę.")
            break
            
        alfa = float(alfa_input)
        proby += 1
        d = oblicz_zasieg(v0, h, alfa)
        
        if margines_lewy <= d <= margines_prawy:
            print(f"\nCel trafiony! Zasięg rzutu wyniósł {d:.2f} metrów.")
            print(f"Całkowita liczba prób: {proby}")
            
            narysuj_trajektorie(v0, h, alfa, d)
            break
        else:
            print(f"Zasięg rzutu: {d:.2f} metrów.")
            print("Nie trafiłeś w cel. Spróbuj ponownie!\n")
            
    except ValueError:
        print("Błąd: Podaj poprawną wartość liczbową!\n")