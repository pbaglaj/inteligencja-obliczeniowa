import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

def test_environment(env_name):
    print(f"--- Uruchamiam: {env_name} ---")
    # render_mode='human' sprawia, że otwiera się okienko z podglądem
    env = gym.make(env_name, render_mode="human")
    observation, info = env.reset()
    
    for _ in range(200): # Uruchamiamy na 200 kroków
        # Losowa akcja z dostępnej puli
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()

# Uruchamiamy gry z wykładu
if __name__ == "__main__":
    # test_environment("LunarLander-v3")
    # test_environment("FrozenLake-v1")
    # 1. Classic Control
    # test_environment("CartPole-v1")
    # 2. Box2D
    # test_environment("BipedalWalker-v3")
    # 3. ToyText
    # test_environment("Taxi-v4")
    # 4. Atari (Wymaga zainstalowania licencji romów)
    # test_environment("PongNoFrameskip-v4")
    # test_environment("ALE/Pong-v5")
    # 5. MuJoCo (Symulacja fizyki 3D)
    test_environment("HalfCheetah-v5")    
  
