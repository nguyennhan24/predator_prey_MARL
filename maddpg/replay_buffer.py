import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, num_agents, max_obs_dim, global_state_dim, action_dim_is_discrete=True):

        self.capacity = capacity # Sức chứa tối đa
        self.ptr = 0    #Con trỏ chỉ vị trí lưu dữ liệu tiếp theo
        self.size = 0  #Số lượng mẫu hiện tại trong buffer
        self.num_agents = num_agents

        self.state_buffer = np.zeros((capacity, global_state_dim), dtype=np.float32) #Lưu trạng thái toàn cục   
        self.next_state_buffer = np.zeros((capacity, global_state_dim), dtype=np.float32) #Lưu trạng thái toàn cục tiếp theo

        self.obs_buffer = np.zeros((capacity, num_agents, max_obs_dim), dtype=np.float32) #Lưu quan sát của từng agent
        self.next_obs_buffer = np.zeros((capacity, num_agents, max_obs_dim), dtype=np.float32) #Lưu quan sát tiếp theo của từng agent

        if action_dim_is_discrete:
            self.action_buffer = np.zeros((capacity, num_agents), dtype=np.int32) #Lưu hành động của từng agent (rời rạc)
        else:
            pass

        self.reward_buffer = np.zeros((capacity, num_agents), dtype=np.float32) #Lưu phần thưởng của từng agent
        self.done_buffer = np.zeros((capacity, num_agents), dtype=np.float32) #


    def add(self, global_state, obs, action, reward, next_global_state, next_obs, done):
        self.state_buffer[self.ptr] = global_state
        self.obs_buffer[self.ptr] = obs
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_global_state
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        """Bước 5: Lấy ngẫu nhiên batch và ép sang PyTorch Tensor"""
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.state_buffer[idxs]).to(device)
        obses = torch.FloatTensor(self.obs_buffer[idxs]).to(device)
        actions = torch.LongTensor(self.action_buffer[idxs]).to(device)
        rewards = torch.FloatTensor(self.reward_buffer[idxs]).to(device)
        next_states = torch.FloatTensor(self.next_state_buffer[idxs]).to(device)
        next_obses = torch.FloatTensor(self.next_obs_buffer[idxs]).to(device)
        dones = torch.FloatTensor(self.done_buffer[idxs]).to(device)

        return states, obses, actions, rewards, next_states, next_obses, dones
    
    