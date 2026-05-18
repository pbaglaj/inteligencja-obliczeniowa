import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from simpful import *

# ==========================================
# 1. INICJALIZACJA SYSTEMU ROZMYTEGO
# ==========================================
FS = FuzzySystem()

# ==========================================
# A) DEFINIOWANIE ZMIENNYCH LINGWISTYCZNYCH
# ==========================================
# 1. KĄT WAHAŃ (Angle) w radianach: od ok. -3.14 (w lewo) do 3.14 (w prawo). 0 to pion.
A_N = TriangleFuzzySet(-3.14, -3.14, 0, term="N") # Negative (Wychylone w lewo)
A_Z = TriangleFuzzySet(-1.0, 0, 1.0, term="Z")    # Zero (Blisko pionu)
A_P = TriangleFuzzySet(0, 3.14, 3.14, term="P")   # Positive (Wychylone w prawo)
FS.add_linguistic_variable("Angle", LinguisticVariable([A_N, A_Z, A_P], concept="Angle (rad)", universe_of_discourse=[-3.14, 3.14]))

# 2. PRĘDKOŚĆ KĄTOWA (Velocity): od -8 do 8
V_N = TriangleFuzzySet(-8.0, -8.0, 0.0, term="N") # Leci w lewo
V_Z = TriangleFuzzySet(-3.0, 0.0, 3.0, term="Z")  # Stoi lub leci wolno
V_P = TriangleFuzzySet(0.0, 8.0, 8.0, term="P")   # Leci w prawo
FS.add_linguistic_variable("Velocity", LinguisticVariable([V_N, V_Z, V_P], concept="Velocity", universe_of_discourse=[-8.0, 8.0]))

# 3. AKCJA: MOMENT OBROTOWY (Torque): od -2.0 do 2.0
T_N = TriangleFuzzySet(-2.0, -2.0, 0.0, term="N") # Kręć w lewo
T_Z = TriangleFuzzySet(-0.5, 0.0, 0.5, term="Z")  # Nie kręć / luz
T_P = TriangleFuzzySet(0.0, 2.0, 2.0, term="P")   # Kręć w prawo
FS.add_linguistic_variable("Torque", LinguisticVariable([T_N, T_Z, T_P], concept="Torque", universe_of_discourse=[-2.0, 2.0]))

# WYŚWIETLENIE WYKRESÓW ZMIENNYCH
print("Generowanie wykresów...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
FS.plot_variable("Angle", ax=ax1)
ax1.set_title("Stan: Kąt")
FS.plot_variable("Velocity", ax=ax2)
ax2.set_title("Stan: Prędkość")
FS.plot_variable("Torque", ax=ax3)
ax3.set_title("Akcja: Siła")
plt.tight_layout()
plt.show() # Zamknij to okno, aby uruchomić grę!

# ==========================================
# B) REGUŁY WNIOSKOWANIA (Z UŻYCIEM "AND")
# ==========================================
# Logika: 
# 1-2: Jeśli jesteśmy mocno wychyleni, pchaj w przeciwną stronę żeby podnieść.
# 3-4: Jeśli jesteśmy blisko pionu (Angle IS Z), patrz na prędkość. Jeśli leci w lewo, kontruj w prawo itd.
# 5: Jeśli jesteśmy w pionie i nie mamy prędkości, nic nie rób.
rules = [
    "IF (Angle IS N) THEN (Torque IS P)",
    "IF (Angle IS P) THEN (Torque IS N)",
    "IF (Angle IS Z) AND (Velocity IS N) THEN (Torque IS P)",
    "IF (Angle IS Z) AND (Velocity IS P) THEN (Torque IS N)",
    "IF (Angle IS Z) AND (Velocity IS Z) THEN (Torque IS Z)"
]
FS.add_rules(rules)

# ==========================================
# C) ODPALENIE SYMULACJI (W KAŻDEJ KLATCE)
# ==========================================
print("\nUruchamianie symulacji Pendulum-v1...")
env = gym.make("Pendulum-v1", render_mode="human")
obs, info = env.reset()

for frame in range(500):
    # 1. Odczyt ze środowiska
    cos_theta = obs[0]
    sin_theta = obs[1]
    velocity = obs[2]
    
    # 2. Przeliczenie sin/cos na kąt w radianach
    angle = np.arctan2(sin_theta, cos_theta)
    
    # 3. Wstrzyknięcie wartości do kontrolera rozmytego
    FS.set_variable("Angle", angle)
    FS.set_variable("Velocity", velocity)
    
    # 4. Wnioskowanie i wyostrzanie (Defuzzification)
    result = FS.Mamdani_inference(["Torque"])
    torque = result["Torque"]
    
    # Gymnasium Box Action Space wymaga podania akcji jako Numpy Array
    action = np.array([torque], dtype=np.float32)
    
    # 5. Wykonanie akcji w środowisku
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Wahadło domyślnie rzuca 'truncated' po 200 krokach
    if terminated or truncated:
        obs, info = env.reset()

env.close()