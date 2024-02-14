from env.pokemon_env import PokemonEnv
from agent.model import DQNAgent
import torch

def main(args) :
    env = PokemonEnv("env_config.yaml")
    infos, img = env.reset()
    agent = DQNAgent("agent_config.yaml")
    agent.target_q_network.load_state_dict(torch.load(args.model_weights))
    agent.target_q_network.to(agent.device)

    agent.frame_stacking.reset(img)
    frames =  torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
    infos = torch.tensor(infos, dtype=torch.float32).unsqueeze(0).to(agent.device)

    for _ in range(10000):
        action = agent.target_q_network(frames, infos).max(1).indices.item()
        print(action)
        infos, img, reward, done  = env.step(action)
        agent.frame_stacking.update_buffer(img)
        frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        infos = torch.tensor(infos, dtype=torch.float32).unsqueeze(0).to(agent.device)

    env.close()

if __name__ == "__main__" :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--model_weights", type=str, default="agent/checkpoints/training_1/target_q_network_6.pth")

    main(parser.parse_args())