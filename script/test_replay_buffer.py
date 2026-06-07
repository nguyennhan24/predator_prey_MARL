# scripts/test_replay_buffer.py (Bước 6)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from mpe2 import simple_tag_v3
from maddpg.config import AGENT_ORDER, BATCH_SIZE
from maddpg.utils import get_env_metadata, dict_obs_to_array, dict_action_to_array, dict_reward_to_array, dict_done_to_array
from maddpg.replay_buffer import ReplayBuffer

def run_test():
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, continuous_actions=False)
    env.reset()

    metadata = get_env_metadata(env, AGENT_ORDER)
    max_obs_dim = metadata['max_obs_dim']
    global_state_dim = metadata['global_state_dim']
    num_agents = len(AGENT_ORDER)

    buffer = ReplayBuffer(capacity=1000, num_agents=num_agents, 
                          max_obs_dim=max_obs_dim, global_state_dim=global_state_dim)

    obs_dict, _ = env.reset()
    
    print("Đang nạp dữ liệu vào Buffer...")
    for step in range(200):
        # Policy ngẫu nhiên
        action_dict = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        next_obs_dict, reward_dict, terminations, truncations, _ = env.step(action_dict)

        # Dict -> Array
        obs_array = dict_obs_to_array(obs_dict, AGENT_ORDER, max_obs_dim)
        next_obs_array = dict_obs_to_array(next_obs_dict, AGENT_ORDER, max_obs_dim)
        action_array = dict_action_to_array(action_dict, AGENT_ORDER)
        reward_array = dict_reward_to_array(reward_dict, AGENT_ORDER)
        done_array = dict_done_to_array(terminations, truncations, AGENT_ORDER)

        # Global state = ghép toàn bộ obs cục bộ thành mảng 1D
        global_state = np.concatenate([obs_dict[agent] for agent in AGENT_ORDER])
        next_global_state = np.concatenate([next_obs_dict[agent] for agent in AGENT_ORDER])

        buffer.add(global_state, obs_array, action_array, reward_array, next_global_state, next_obs_array, done_array)

        obs_dict = next_obs_dict
        
        if any(terminations.values()) or any(truncations.values()):
            obs_dict, _ = env.reset()

    print(f"\nLấy mẫu Batch Size: {BATCH_SIZE}")
    states, obses, actions, rewards, next_states, next_obses, dones = buffer.sample(BATCH_SIZE)

    print("States shape:      ", states.shape)
    print("Obses shape:       ", obses.shape)
    print("Actions shape:     ", actions.shape)
    print("Rewards shape:     ", rewards.shape)
    print("Next States shape: ", next_states.shape)
    print("Next Obses shape:  ", next_obses.shape)
    print("Dones shape:       ", dones.shape)

if __name__ == "__main__":
    run_test()