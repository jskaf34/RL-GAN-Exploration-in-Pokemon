import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayBuffer, FrameStacker
import random
from pokemon_env import PokemonEnv
import math
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, action_size=7, input_channels=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Input shape is (B, 4, 144, 160)
        self.fc1 = nn.Linear(32 * 16 * 18 + 4, 256)
        self.fc2 = nn.Linear(256, action_size)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, img, infos):
        x = self.relu(self.conv1(img))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.cat([x, infos], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# DQN agent
class DQNAgent:
    def __init__(self, action_size=7, learning_rate=0.0001, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000):
        self.action_size = action_size
        self.q_network = QNetwork()
        self.target_q_network = QNetwork()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, amsgrad=True)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.frame_stacking = FrameStacker(num_frames=4, frame_shape=(144, 160))
        self.step = 0
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            math.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.q_network(*state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.action_size - 1)]], device=self.device, dtype=torch.long)

    def update_model(self, batch_size):
        if len(self.replay_buffer.memory) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = self.replay_buffer.Transition(*zip(*transitions))

        stacked_frames_batch = torch.cat(batch.stacked_frames)
        state_infos_batch = torch.cat(batch.state_infos)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        next_stacked_frames_batch = torch.cat(batch.next_stacked_frames)
        next_state_infos_batch = torch.cat(batch.next_state_infos)

        current_q_values = self.q_network(stacked_frames_batch, state_infos_batch).gather(1, action_batch.unsqueeze(0))
        next_q_values = self.target_q_network(next_stacked_frames_batch, next_state_infos_batch).max(1).values.view(1,32).squeeze(0)

        expected_q_values = reward_batch + self.gamma * next_q_values

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()


def preprocess_img(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Training loop
def train(env, agent, num_episodes=1000, batch_size=32):
    for episode in range(num_episodes):
        infos, img = env.reset()
        state_infos = torch.tensor(list(infos.values()), dtype=torch.float32).unsqueeze(0)
        agent.frame_stacking.reset(preprocess_img(img))
        stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        total_reward = 0

        while True:
            action = agent.select_action((stacked_frames, state_infos))[0][0].item()
            next_state_infos, next_state_img, reward, done = env.step(action)
            next_state_infos = torch.tensor(list(next_state_infos.values()), dtype=torch.float32).unsqueeze(0)
            agent.frame_stacking.update_buffer(preprocess_img(next_state_img))
            next_stacked_frames = torch.tensor(agent.frame_stacking.stack_frames(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            agent.replay_buffer.push(stacked_frames, state_infos, torch.tensor(action).unsqueeze(0), next_stacked_frames, next_state_infos, torch.tensor(reward).unsqueeze(0), done)
            agent.update_model(batch_size)
            total_reward += reward

            if done:
                break

            state_infos = next_state_infos
            stacked_frames = next_stacked_frames

        agent.update_target_network()
        agent.decay_epsilon()

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Example usage
env = PokemonEnv('jeu/PokemonRed.gb')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent()

train(env, agent, num_episodes=1000)