import numpy as np
import torch
import torch.nn.functional as F

def get_env_metadata(env, agent_order):
    obs_dims = {}      # Sửa: Thêm 's' để thể hiện đây là dictionary
    action_dims = {}   # Sửa: Thêm 's' để thể hiện đây là dictionary
    global_state_dim = 0

    for agent in agent_order:
        # Xử lý Observation
        obs_shape = env.observation_space(agent).shape
        obs_dim = obs_shape[0] if len(obs_shape) > 0 else 1
        obs_dims[agent] = obs_dim

        # Sửa: obs_dim hiện tại là 1 con số (int), không thể dùng obs_dim[agent]
        global_state_dim += obs_dim 

        # Xử lý Action
        action_shape = env.action_space(agent)

        if hasattr(action_shape, 'n'):
            action_dims[agent] = action_shape.n
        else:
            action_dims[agent] = action_shape.shape[0]

        # Xóa dòng action_dim[agent] = action_dim vì nó tạo ra vòng lặp đệ quy gán chính cái dict vào trong dict.

    # Sửa: Lấy max từ values của dictionary 'obs_dims', không phải số nguyên 'obs_dim'
    max_obs_dim = max(obs_dims.values())

    return {
        'obs_dims': obs_dims,       # Trả về đúng tên biến dictionary
        'action_dims': action_dims, # Trả về đúng tên biến dictionary
        'global_state_dim': global_state_dim,
        'max_obs_dim': max_obs_dim
    }

def dict_obs_to_array(obs_dict, agent_order, max_obs_dim):
    num_agents = len(agent_order)
    obs_array = np.zeros((num_agents, max_obs_dim), dtype=np.float32)

    for i, agent in enumerate(agent_order):
        obs = obs_dict[agent]
        obs_array[i, :len(obs)] = obs

    return obs_array

def dict_action_to_array(action_dict, agent_order):
    actions = [action_dict[agent] for agent in agent_order]
    return np.array(actions, dtype=np.float32)

def dict_reward_to_array(reward_dict, agent_order):
    rewards = [reward_dict[agent] for agent in agent_order]
    return np.array(rewards, dtype=np.float32)

# Sửa: Đổi tham số đầu vào thành terminations và truncations thay vì done_dict
def dict_done_to_array(terminations, truncations, agent_order):
    dones = []

    for agent in agent_order:
        # Code cũ của bạn gọi terminations và truncations nhưng hàm lại không nhận hai biến này
        is_done = terminations[agent] or truncations[agent]
        dones.append(float(is_done))

    return np.array(dones, dtype=np.float32)

def one_hot_encode_actions(actions, num_actions, device):
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions, dtype=torch.long, device=device)
    
    actions = actions.view(-1)
    
    return F.one_hot(actions, num_classes=num_actions).float()