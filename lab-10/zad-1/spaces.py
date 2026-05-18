import gymnasium as gym

def check_spaces(env_name, space_description):
    env = gym.make(env_name)
    print(f"\nGra: {env_name} | {space_description}")
    print(f"Przestrzeń stanów (Observation): {env.observation_space}")
    print(f"Przestrzeń akcji (Action): {env.action_space}")
    env.close()

# 1. Stan gry i zestaw akcji są dyskretne:
# Przykład: FrozenLake-v1. Stan to numer pola (0-15), a akcja to kierunek (0-3).
check_spaces("FrozenLake-v1", "TYP: Stan DYSKRETNY, Akcje DYSKRETNE")

# 2. Stan gry jest ciągły, ale zestaw akcji jest dyskretny:
# Przykład: CartPole-v1. Stan to liczby (kąt wózka, prędkość), akcja to 0 (w lewo) lub 1 (w prawo).
check_spaces("CartPole-v1", "TYP: Stan CIĄGŁY (Box), Akcje DYSKRETNE (Discrete)")

# 3. Stan gry i zestaw akcji są ciągłe:
# Przykład: Pendulum-v1. Stan to kąt i prędkość wahadła (ciągłe), akcja to siła obrotowa w zakresie od -2.0 do 2.0 (ciągła).
check_spaces("Pendulum-v1", "TYP: Stan CIĄGŁY (Box), Akcje CIĄGŁE (Box)")