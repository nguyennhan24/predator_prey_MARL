import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim=16, action_dim=5, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logit = self.out(x)
        return logit

class Critic(nn.Module):
    def __init__(self, global_state_dim=62, action_dim=20, hidden_dim=256):
        super().__init__()
        input_dim = global_state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, actions_one_hot_flat):
        x = torch.cat([state, actions_one_hot_flat], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value