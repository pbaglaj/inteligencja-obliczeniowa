import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(500):
    # W CartPole observation[2] to kąt nachylenia kija
    angle = observation[2]
    
    # NASZA LOGIKA (zamiast env.action_space.sample())
    if angle > 0:
        action = 1  # Wózek w prawo
    else:
        action = 0  # Wózek w lewo
        
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()