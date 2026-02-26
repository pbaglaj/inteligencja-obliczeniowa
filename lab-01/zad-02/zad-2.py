import math
import random
import matplotlib.pyplot as plt
# do zrobienia wykresu potrzebujemy listy kątów i odpowiadających im zasięgów

odleglosc_celu = random.randint(50, 340)
margines_lewy = odleglosc_celu - 5
margines_prawy = odleglosc_celu + 5 
print(f"Odległość celu: {odleglosc_celu} metrów")

h = 100
v0 = 50

def oblicz_zasieg(v0, h, alfa_stopnie):
    alfa = math.radians(alfa_stopnie)
    zasieg = ((v0 * math.sin(alfa)) + (math.sqrt((v0**2) * (math.sin(alfa)**2) + 2 * h * 9.81))) * ((v0 * math.cos(alfa)) / 9.81)
    return max(0.0, zasieg)

alfa = float(input("Podaj kąt alfa (w stopniach): "))
d = oblicz_zasieg(v0, h, alfa)

while True:    
  if (d >= margines_lewy and d <= margines_prawy):
      print("Trafiłeś w cel!")
      break
  else:
      print(f"Zasięg rzutu: {d:.2f} metrów")
      print("Nie trafiłeś w cel. Spróbuj ponownie!")
      alfa = float(input("Podaj kąt alfa (w stopniach): "))
      d = oblicz_zasieg(v0, h, alfa)
