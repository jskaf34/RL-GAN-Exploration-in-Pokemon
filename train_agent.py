import torch
from model import DQNAgent
from datetime import datetime

def train(env, agent, num_episodes, batch_size, save_dir):
    agent.q_network.to(agent.device)
    agent.target_q_network.to(agent.device)
    for episode in range(num_episodes):
        infos, img = env.reset()
        state_infos = torch.tensor(list(infos.values()), dtype=torch.float32).unsqueeze(0).to(agent.device)
        agent.frame_stacking.reset(img)
        stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        total_reward = 0

        while True:
            action = agent.select_action((stacked_frames, state_infos))[0][0].item()
            next_state_infos, next_state_img, reward, done = env.step(action)
            next_state_infos = torch.tensor(list(next_state_infos.values()), dtype=torch.float32).unsqueeze(0).to(agent.device)
            agent.frame_stacking.update_buffer(next_state_img)
            next_stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(agent.device)
            agent.replay_buffer.push(stacked_frames, state_infos, torch.tensor(action).unsqueeze(0).to(agent.device), next_stacked_frames, next_state_infos, torch.tensor(reward).unsqueeze(0).to(agent.device), done)
            agent.update_model(batch_size)
            total_reward += reward

            if done:
                break

            state_infos = next_state_infos
            stacked_frames = next_stacked_frames

            if env.nb_step % 10000 == 0 :
                print(env.nb_step, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        agent.update_target_network()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        if episode % 5 == 0:
            torch.save(agent.q_network.state_dict(), f"{save_dir}/q_network_{episode + 1}.pth")
            torch.save(agent.target_q_network.state_dict(), f"{save_dir}/target_q_network_{episode + 1}.pth")

def main(args):
    from pokemon_env import PokemonEnv
    import os

    env = PokemonEnv('jeu/PokemonRed.gb', render_reward=False)
    agent = DQNAgent()
    print("Using device : ", agent.device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dirs = [d for d in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, d))]

    # Create a new directory with a name based on the count
    new_dir_name = f"training_{len(dirs) + 1}"
    os.makedirs(os.path.join(args.save_dir, new_dir_name))

    train(env, agent, args.num_episodes, args.batch_size, os.path.join(args.save_dir, new_dir_name))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--save_dir', type=str, default='checkpoints/')
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument("-ne", "--num_episodes", type=int, default=1000)

    main(parser.parse_args())