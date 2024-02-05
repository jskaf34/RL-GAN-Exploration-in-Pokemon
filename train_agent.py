import torch
from model import DQNAgent, DDQNAgent
import math
from tqdm import tqdm

def train(env, agent, num_episodes, batch_size, save_dir, from_pretrained):
    agent.q_network.to(agent.device)
    agent.target_q_network.to(agent.device)

    for episode in range(num_episodes):
        if episode <= 10 and from_pretrained:
            agent.step += env.max_step / env.action_freq
            continue 
        
        if episode % 5 == 0:
            env.video_path = f"{save_dir}/video_{episode + 1}.mp4"
        else: 
            env.video_path = f"{save_dir}/video.mp4"
        infos, img = env.reset()
        state_infos = torch.tensor(infos, dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.frame_stacking.reset(img)
        stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        total_reward = 0

        for _ in tqdm(range(env.max_step)):
            action = agent.select_action((stacked_frames, state_infos))
            next_state_infos, next_state_img, reward, done = env.step(action)
            next_state_infos = torch.tensor(next_state_infos, dtype=torch.float32).unsqueeze(0).to(agent.device)
            agent.frame_stacking.update_buffer(next_state_img)
            next_stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
            agent.replay_buffer.push(stacked_frames, state_infos, torch.tensor(action).unsqueeze(0).to(agent.device), next_stacked_frames, next_state_infos, torch.tensor(reward).unsqueeze(0).to(agent.device), done)
            agent.update_model(batch_size)
            agent.update_target_network()
            total_reward += reward

            state_infos = next_state_infos
            stacked_frames = next_stacked_frames

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon_end + (agent.epsilon - agent.epsilon_end) * math.exp(-1. * agent.step / agent.epsilon_decay)}")

        if episode % 5 == 0:
            torch.save(agent.q_network.state_dict(), f"{save_dir}/q_network_{episode + 1}.pth")
            torch.save(agent.target_q_network.state_dict(), f"{save_dir}/target_q_network_{episode + 1}.pth")
            with open(f"{save_dir}/state_{episode + 1}.state", "wb") as state_file:
                env.pyboy.save_state(state_file)

def main(args):
    from pokemon_env import PokemonEnv
    import os

    env = PokemonEnv('env_config.yaml')
    agent = DQNAgent("agent_config.yaml")
    # agent = DDQNAgent("agent_config.yaml")

    if args.from_pretrained :
        agent.q_network.load_state_dict(torch.load("checkpoints/training_1/q_network_11.pth"))
        agent.target_q_network.load_state_dict(torch.load("checkpoints/training_1/target_q_network_11.pth"))

    print("Using device : ", agent.device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    dirs = [d for d in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, d))]
    new_dir_name = f"training_{len(dirs) + 1}"
    os.makedirs(os.path.join(args.save_dir, new_dir_name))

    train(env, agent, args.num_episodes, args.batch_size, os.path.join(args.save_dir, new_dir_name), args.from_pretrained)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--save_dir', type=str, default='checkpoints/')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument("-ne", "--num_episodes", type=int, default=500)
    parser.add_argument('--from_pretrained', action='store_true')
    parser.add_argument("-na", '--nb_action', type=int, default=6)

    main(parser.parse_args())