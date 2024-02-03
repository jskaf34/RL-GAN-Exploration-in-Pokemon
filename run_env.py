from pokemon_env import PokemonEnv

# Usage example:
env = PokemonEnv("jeu/PokemonRed.gb", emulation_speed=2, render_reward=True)
observation = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    agent_state, screen, reward  = env.step(action)

env.close()