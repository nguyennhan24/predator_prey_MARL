import sys
import os
import torch
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from maddpg.agent import MADDPGAgent  # Bỏ dấu chấm '..' ở đầu đi

def run_sanity_check():
    device = "cpu"
    batch_size = 1024 #
    num_agents = 4 #
    obs_dim = 16 # 
    global_state_dim = 62 # 
    action_dim = 5 #

    # 1. Tạo 4 Agents 
    agents = [MADDPGAgent(
        agent_id=i, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        global_state_dim=global_state_dim, 
        total_action_dim=num_agents * action_dim
    ) for i in range(num_agents)]

    # 2. Tạo Mock Batch từ Replay Buffer (Dữ liệu giả lập) 
    states = torch.randn(batch_size, global_state_dim) # 
    obs = torch.randn(batch_size, num_agents, obs_dim) #
    # Giả lập actions đang ở dạng discrete index (Lưu ý: dùng dtype torch.long)
    actions = torch.randint(0, action_dim, (batch_size, num_agents)) #

    # Xử lý: [B, 4] -> One-hot [B, 4, 5] -> Flatten [B, 20] 
    actions_one_hot = F.one_hot(actions, num_classes=action_dim).float() # 
    actions_flat = actions_one_hot.reshape(batch_size, num_agents * action_dim) #

    print("--- KIỂM TRA BATCH SHAPE ---")
    print(f"states:          {states.shape}") # Kỳ vọng: [1024, 62] 
    print(f"obs:             {obs.shape}") # Kỳ vọng: [1024, 4, 16] 
    print(f"actions:         {actions.shape}") # Kỳ vọng: [1024, 4] 
    print(f"actions_one_hot: {actions_one_hot.shape}") # Kỳ vọng: [1024, 4, 5] 
    print(f"actions_flat:    {actions_flat.shape}") # Kỳ vọng: [1024, 20]
    print("----------------------------\n")

    # 3. Test Forward Pass cho 4 agents
    for i, agent in enumerate(agents):
        obs_i = obs[:, i, :] # Lấy đúng obs của agent i 
        logits_i = agent.actor(obs_i) # Test Actor
        q_value = agent.critic(states, actions_flat) # Test Critic

        print(f"Agent {i}:")
        print(f"  obs_i:    {obs_i.shape}") #
        print(f"  logits:   {logits_i.shape}") # Kỳ vọng: [1024, 5]
        print(f"  q_value:  {q_value.shape}\n") # Kỳ vọng: [1024, 1] 

    print("Target networks initialized successfully.")

if __name__ == "__main__":
    run_sanity_check()