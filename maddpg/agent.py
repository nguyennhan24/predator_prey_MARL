import torch
import torch.optim as optim
import numpy as np
import sys
import os
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from maddpg.network import Actor, Critic
from maddpg.utils import one_hot_encode_actions

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class MADDPGAgent:
    def __init__(self, agent_id, obs_dim=16, action_dim=5, global_state_dim=5,
                 total_action_dim=20, actor_hidden_dim=128, critic_hidden_dim=256, 
                 actor_lr=1e-3, critic_lr=1e-3, device='cpu'):
        
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.device = device

        self.actor = Actor(obs_dim, action_dim, actor_hidden_dim).to(device)
        self.critic = Critic(global_state_dim, total_action_dim, critic_hidden_dim).to(device)

        self.target_actor = Actor(obs_dim, action_dim, actor_hidden_dim).to(device)
        self.target_critic = Critic(global_state_dim, total_action_dim, critic_hidden_dim).to(device)

        self.hard_update_targets()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, explore=True, epsilon=0.1):
        self.actor.eval()

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(obs_tensor)

            if explore:
                gumbel_action = F.gumbel_softmax(logits, tau=1.0, hard=True)
                action_index = torch.argmax(gumbel_action, dim=-1)
            
            else:
                action_index = torch.argmax(logits, dim=-1)

        action_index = action_index.item()

        return action_index

    def hard_update_targets(self):
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def soft_update_targets(self, tau):
        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau) 

    def update(self, agents, replay_buffer, batch_size, gamma=0.95):
        """
        agents: Danh sách tất cả các object MADDPGAgent trong môi trường.
        """
        # 1. Lấy mẫu từ Replay Buffer
        global_states, states, actions, rewards, next_global_states,  next_states, dones = replay_buffer.sample(batch_size, device=self.device)

        # ==========================================
        # A. CẬP NHẬT CRITIC
        # ==========================================
        target_next_actions = []
        with torch.no_grad():
            for j, other_agent in enumerate(agents):
                # Trích xuất next_obs của agent thứ j
                next_obs_j = next_states[:, j, :]
                # Target Actor dùng Softmax để lấy phân phối xác suất mềm (kỳ vọng)
                next_logits = other_agent.target_actor(next_obs_j)
                next_action = F.softmax(next_logits, dim=-1)
                target_next_actions.append(next_action)
        
        target_actions_cat = torch.cat(target_next_actions, dim=1)
        next_states_cat = next_states.view(batch_size, -1)

        # Tính Target Q cho CHÍNH AGENT NÀY (self.agent_id)
        with torch.no_grad():
            target_q_next = self.target_critic(next_states_cat, target_actions_cat)
            target_q = rewards[:, self.agent_id].unsqueeze(1) + gamma * target_q_next * (1 - dones[:, self.agent_id].unsqueeze(1))

        # Chuyển đổi actions ID hiện tại từ buffer sang one-hot cho tất cả agents
        current_actions = []
        for j, other_agent in enumerate(agents):
            one_hot_a = one_hot_encode_actions(actions[:, j], other_agent.action_dim, self.device)
            current_actions.append(one_hot_a)
            
        current_actions_cat = torch.cat(current_actions, dim=1)
        states_cat = states.view(batch_size, -1)

        # Critic dự đoán Q-value hiện tại
        expected_q = self.critic(states_cat, current_actions_cat)

        # Loss MSE và lan truyền ngược
        critic_loss = F.mse_loss(expected_q, target_q)

        self.critic_optimizer.zero_grad() #xóa đọa hàm về 0 để tính cho lứa mới
        critic_loss.backward() # Backpropagation
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Tránh bùng nổ gradient
        self.critic_optimizer.step()

        # ==========================================
        # B. CẬP NHẬT ACTOR
        # ==========================================
        actor_actions = []
        for j, other_agent in enumerate(agents):
            obs_j = states[:, j, :]
            
            if j == self.agent_id:
                # Nếu là agent của chính mình -> Lấy gradient qua Gumbel-Softmax
                logits = self.actor(obs_j)
                action_j = F.gumbel_softmax(logits, tau=1.0, hard=True)
            else:
                # Nếu là agent khác -> Mượn action làm đầu vào cho Critic, không lấy gradient
                with torch.no_grad():
                    logits = other_agent.actor(obs_j)
                    action_j = F.gumbel_softmax(logits, tau=1.0, hard=True)
                    
            actor_actions.append(action_j)
        
        actor_actions_cat = torch.cat(actor_actions, dim=1)

        # Actor Loss là giá trị Q âm (để tối đa hóa Q)
        actor_loss = -self.critic(states_cat, actor_actions_cat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # RẤT QUAN TRỌNG: Phải update Target Network thì AI mới học được!
        self.soft_update_targets(tau=0.01)