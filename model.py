import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayBuffer, FrameStacker
import random
import math
import yaml

class QNetwork(nn.Module):
    def __init__(self, action_size=7, input_channels=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Input shape is (B, 4, 100, 100)
        self.fc1 = nn.Linear(32 * 11 * 11 + 4, 256)
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
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.action_size = config["action_size"]
        self.q_network = QNetwork(action_size=config["action_size"])
        self.target_q_network = QNetwork(action_size=config["action_size"])
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=config["learning_rate"], amsgrad=True)
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.frame_stacking = FrameStacker(num_frames=4, frame_shape=config["image_shape"])
        self.step = 0
        self.device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            math.exp(-1. * self.step / self.epsilon_decay)
        self.step += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.q_network(*state).max(1).indices.item()
        else:
            return random.randint(0, self.action_size - 1)

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
        next_q_values = self.target_q_network(next_stacked_frames_batch, next_state_infos_batch).max(1).values

        expected_q_values = reward_batch + self.gamma * next_q_values

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values.unsqueeze(0))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        target_net_state_dict = self.target_q_network.state_dict()
        q_net_state_dict = self.q_network.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_q_network.load_state_dict(target_net_state_dict)