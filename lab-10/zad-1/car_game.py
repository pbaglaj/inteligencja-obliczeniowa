import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset()

for _ in range(500):
    # W MountainCar observation[1] to aktualna prędkość (velocity)
    velocity = observation[1]
    
    # NASZA LOGIKA
    if velocity > 0:
        action = 2  # Przyspieszaj w prawo
    elif velocity < 0:
        action = 0  # Przyspieszaj w lewo (cofaj)
    else:
        action = 1  # Nic nie rób
        
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()