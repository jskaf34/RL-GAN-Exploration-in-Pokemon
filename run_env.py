from pokemon_env import PokemonEnv
import time

# Usage example:
env = PokemonEnv("env_config.yaml")
observation = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    print(action)
    agent_state, frames, reward, _  = env.step(action)
    time.sleep(1)

env.close()