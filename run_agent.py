from pokemon_env import PokemonEnv
from model import DQNAgent
import torch

env = PokemonEnv("jeu/PokemonRed.gb", emulation_speed=0, render_reward=True, render_view=True)
infos, img = env.reset()
agent = DQNAgent()
agent.q_network.load_state_dict(torch.load('checkpoints/training_2/q_network_6.pth'))
agent.target_q_network.load_state_dict(torch.load('checkpoints/training_2/target_q_network_6.pth'))

agent.q_network.to(agent.device)
agent.target_q_network.to(agent.device)

agent.frame_stacking.reset(img)
frames =  torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
infos = torch.tensor(list(infos.values()), dtype=torch.float32).unsqueeze(0).to(agent.device)

for _ in range(10000):
    action = agent.q_network(frames, infos).max(1).indices.view(1, 1)[0][0].item()
    infos, img, reward, done  = env.step(action)
    agent.frame_stacking.update_buffer(img)
    frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
    infos = torch.tensor(list(infos.values()), dtype=torch.float32).unsqueeze(0).to(agent.device)

env.close()