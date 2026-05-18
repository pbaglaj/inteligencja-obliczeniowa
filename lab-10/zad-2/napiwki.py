from simpful import *
import matplotlib.pyplot as plt

# 1. Inicjalizacja systemu rozmytego
FS = FuzzySystem()

# ==========================================
# 2. DEFINIOWANIE ZMIENNYCH LINGWISTYCZNYCH
# ==========================================

# Zmienna wejściowa: Jakość obsługi (Service) w skali 0-10
S_1 = TriangleFuzzySet(0, 0, 5, term="poor")        # Słaba
S_2 = TriangleFuzzySet(0, 5, 10, term="good")       # Dobra
S_3 = TriangleFuzzySet(5, 10, 10, term="excellent") # Świetna
FS.add_linguistic_variable("Service", LinguisticVariable([S_1, S_2, S_3], concept="Service quality", universe_of_discourse=[0,10]))

# Zmienna wejściowa: Jakość jedzenia (Food) w skali 0-10
F_1 = TriangleFuzzySet(0, 0, 10, term="rancid")     # Okropne
F_2 = TriangleFuzzySet(0, 10, 10, term="delicious") # Wyśmienite
FS.add_linguistic_variable("Food", LinguisticVariable([F_1, F_2], concept="Food quality", universe_of_discourse=[0,10]))

# Zmienna wyjściowa: Napiwek (Tip) w skali 0-30 (%)
T_1 = TriangleFuzzySet(0, 0, 13, term="low")        # Mały napiwek
T_2 = TriangleFuzzySet(0, 13, 25, term="medium")    # Średni napiwek
T_3 = TriangleFuzzySet(13, 25, 30, term="high")     # Duży napiwek
FS.add_linguistic_variable("Tip", LinguisticVariable([T_1, T_2, T_3], concept="Tip percentage", universe_of_discourse=[0,30]))

# ==========================================
# 3. REGUŁY ROZMYTE (FUZZY RULES)
# ==========================================
# Reguły zapisujemy w sposób zbliżony do języka naturalnego
RULE1 = "IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS low)"
RULE2 = "IF (Service IS good) THEN (Tip IS medium)"
RULE3 = "IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS high)"

# Dodajemy reguły do systemu
FS.add_rules([RULE1, RULE2, RULE3])

# ==========================================
# 4. WYŚWIETLANIE WYKRESÓW
# ==========================================
print("Generowanie wykresów zmiennych lingwistycznych...")
# Simpful zazwyczaj nie pauzuje kodu do wyświetlenia jak matplotlib, 
# więc aby na 100% je zachować, można je zapisać do plików lub narysować obok siebie.
# Wykorzystamy matplotlib dla interaktywnego okienka z podglądem.

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
FS.plot_variable("Service", ax=ax1)
ax1.set_title("Zmienna: Obsługa")
FS.plot_variable("Food", ax=ax2)
ax2.set_title("Zmienna: Jedzenie")
FS.plot_variable("Tip", ax=ax3)
ax3.set_title("Zmienna: Napiwek")

plt.tight_layout()
plt.show()  # To zatrzyma skrypt i otworzy okienko z wykresami

# ==========================================
# 5. TESTOWANIE DZIAŁANIA KONTROLERA
# ==========================================
print("\n--- TESTOWANIE NAPIWKÓW (Po zamknięciu wykresów) ---")

# Kilka scenariuszy (Jedzenie, Obsługa)
test_cases = [
    {"Food": 10.0, "Service": 10.0, "opis": "Idealnie (Świetna obsługa, wyśmienite jedzenie)"},
    {"Food": 1.0,  "Service": 1.0,  "opis": "Tragedia (Słaba obsługa, okropne jedzenie)"},
    {"Food": 5.0,  "Service": 5.0,  "opis": "Przeciętnie (Dobra obsługa, średnie jedzenie)"},
    {"Food": 9.0,  "Service": 2.0,  "opis": "Nierówno (Słaba obsługa, pyszne jedzenie)"}
]

for test in test_cases:
    # Wstrzyknięcie wartości na wejście
    FS.set_variable("Food", test["Food"])
    FS.set_variable("Service", test["Service"])
    
    # Uruchomienie wnioskowania i wyostrzanie (defuzzification)
    # Zwracany jest słownik z kluczem będącym nazwą zmiennej wyjściowej ("Tip")
    wynik = FS.Mamdani_inference(["Tip"])
    procent_napiwku = wynik["Tip"]
    
    print(f"[{test['opis']}]")
    print(f" --> Jedzenie: {test['Food']}/10, Obsługa: {test['Service']}/10")
    print(f" --> SUGEROWANY NAPIWEK: {procent_napiwku:.1f}%\n")