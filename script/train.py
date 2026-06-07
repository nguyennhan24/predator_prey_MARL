import torch
from mpe2 import simple_tag_v3

# Giả sử bạn có class ReplayBuffer trong buffer.py
import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from maddpg.replay_buffer import ReplayBuffer
from maddpg.agent import MADDPGAgent

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # KHỞI TẠO: Không bật đồ họa (render_mode=None) để train siêu tốc trên GPU
    env = simple_tag_v3.parallel_env(render_mode=None, continuous_actions=False)
    env.reset()
    
    # Khai báo bộ não (maddpg) và bộ nhớ (buffer)
    maddpg_agents = []
    for i in range(env.num_agents):
        # Lưu ý: Các tham số obs_dim, global_state_dim có thể cần điều chỉnh đúng với môi trường của bạn
        agent = MADDPGAgent(agent_id=i, obs_dim=16, action_dim=5, 
                            global_state_dim=16*env.num_agents, 
                            total_action_dim=5*env.num_agents, device=device)
        maddpg_agents.append(agent)    
    buffer = ReplayBuffer(capacity=100000, num_agents=env.num_agents, max_obs_dim=16, global_state_dim=16*env.num_agents)
    
    batch_size = 256 
    episodes = 2000
    reward_history = []
    
    print("Bắt đầu huấn luyện...")
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Một hiệp chơi giới hạn 25 bước
        for step in range(25): 
            for a in env.possible_agents:
                if len(obs[a]) < 16:
                    obs[a] = np.pad(obs[a], (0, 16 - len(obs[a])), 'constant')            
            actions = {}
            
            # 1. Từng agent chọn hành động
            for agent_id, agent_name in enumerate(env.possible_agents):
                # Hàm select_action này của bạn đã có cơ chế Explore (Epsilon-greedy)
                action = maddpg_agents[agent_id].select_action(obs[agent_name], explore=True)
                actions[agent_name] = action
                
            # 2. Cả đám hành động trong môi trường
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            for a in env.possible_agents:
                if len(next_obs[a]) < 16:
                    next_obs[a] = np.pad(next_obs[a], (0, 16 - len(next_obs[a])), 'constant')

            # 3. Lưu toàn bộ 1 khung hình (trạng thái, hành động, phần thưởng...) vào bộ nhớ
            # (Bạn cần map dict ra list/array tùy cách bạn viết hàm push của ReplayBuffer)
            obs_list = np.array([obs[agent] for agent in env.possible_agents])
            next_obs_list = np.array([next_obs[agent] for agent in env.possible_agents])
            action_list = np.array([actions[agent] for agent in env.possible_agents])
            reward_list = np.array([rewards[agent] for agent in env.possible_agents])
            done_list = np.array([terminations[agent] for agent in env.possible_agents])            
            
            global_state = obs_list.flatten() # Giả sử global state là concat của obs cục bộ
            next_global_state = next_obs_list.flatten()
            
            buffer.add(global_state, obs_list, action_list, reward_list, next_global_state, next_obs_list, done_list)
            
            obs = next_obs
            episode_reward += sum(rewards.values()) # Cộng tổng phần thưởng để in ra xem
            
            # 4. Khi bộ nhớ đủ lớn, bắt đầu rút kinh nghiệm (Gọi hàm update!)
            if buffer.size > batch_size and step % 5 == 0: # Cứ 5 bước thì update một lần
                for agent in maddpg_agents:
                    agent.update(maddpg_agents, buffer, batch_size)
                
            if all(terminations.values()) or all(truncations.values()):
                break
                
        reward_history.append(episode_reward)
        # In log ra terminal để theo dõi
        print(f"Episode {ep+1}/{episodes} | Total Reward: {episode_reward:.2f}")
        
        # LƯU CHỐT (Checkpoint): Cứ 500 hiệp thì save model lại 1 lần
        if (ep + 1) % 500 == 0:
            for i, agent in enumerate(maddpg_agents):
                torch.save(agent.actor.state_dict(), f"actor_agent_{i}.pth")
            print("Đã lưu Checkpoint!")

    np.save("rewards.npy", np.array(reward_history))
    print("Đã lưu lịch sử phần thưởng ra file rewards.npy")

if __name__ == "__main__":
    train()