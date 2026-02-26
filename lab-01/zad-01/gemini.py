import math
from datetime import date

def oblicz_biorytm(t, cykl):
    # Wzór: y = sin((2 * pi / cykl) * t)
    return math.sin((2 * math.pi / cykl) * t)

def interpretuj_wynik(nazwa, wynik, t, cykl):
    print(f"--- {nazwa} ---")
    print(f"Wynik: {wynik:.4f}")
    
    if wynik > 0.5:
        print(f"Gratulacje! Twój poziom {nazwa.lower()} jest dzisiaj bardzo wysoki!")
    elif wynik < -0.5:
        print(f"Dzisiaj masz słabszy dzień pod względem {nazwa.lower()}. Głowa do góry!")
        
        # Sprawdzanie trendu na jutro (t + 1)
        wynik_jutro = oblicz_biorytm(t + 1, cykl)
        if wynik_jutro > wynik:
            print("Nie martw się. Jutro będzie lepiej!")
        else:
            print("Pamiętaj, że każdy cykl w końcu odbija się od dna.")
    else:
        print(f"Twój poziom {nazwa.lower()} jest dziś w normie.")
    print()

def main():
    # Pobieranie danych od użytkownika
    imie = input("Podaj swoje imię: ")
    rok = int(input("Podaj rok urodzenia (np. 1995): "))
    miesiac = int(input("Podaj miesiąc urodzenia (1-12): "))
    dzien = int(input("Podaj dzień urodzenia: "))

    data_urodzenia = date(rok, miesiac, dzien)
    dzisiaj = date.today()
    
    # Obliczanie różnicy dni (t)
    t = (dzisiaj - data_urodzenia).days

    print(f"\nWitaj {imie}!")
    print(f"Dzisiaj jest Twój {t}. dzień życia.\n")

    # Definicja cykli: (Nazwa, Dni)
    cykle = [
        ("Fizyczny", 23),
        ("Emocjonalny", 28),
        ("Intelektualny", 33)
    ]

    # Obliczenia i wyświetlanie wyników
    for nazwa, dni in cykle:
        wynik = oblicz_biorytm(t, dni)
        interpretuj_wynik(nazwa, wynik, t, dni)

if __name__ == "__main__":
    main()

# Czas wykonania: 5 minut