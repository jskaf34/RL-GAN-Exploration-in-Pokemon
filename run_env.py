from pokemon_env import PokemonBlueEnv
from time import sleep

# Usage example:
env = PokemonBlueEnv("jeu/PokemonRed.gb", emulation_speed=2, render_reward=True)
observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    agent_state, screen, reward  = env.step(action)

env.close()