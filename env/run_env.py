from pokemon_env import PokemonEnv
from PIL import Image

# Usage example:
env = PokemonEnv("../configs/env_config.yaml")
observation = env.reset()
save = False

for i in range(10000):
    action = input("Action ?")
    if action == "save" :
        save = True
        action = input("Action 2 ?")

    if action == "stop" :
        save=False
        action = input("Action 3 ?")

    action = int(action)
    agent_state, frames, reward, _  = env.step(action)
    print("Reward: ", reward, "Agent state: ", agent_state, "Frames: ", frames.shape)
    if save :
        Image.fromarray(frames).save(f"fight_frames/{i}.png")

env.close()