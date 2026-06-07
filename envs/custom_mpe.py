from mpe2 import simple_tag_v3
import numpy as np

class CustomPredatorPreyEnv:
    def __init__(self, num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, render_mode=None, 
                 alpha=0.1, collision_reward=10, beta=5 ):
        
        
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.num_obstacles = num_obstacles
        self.max_cycles = max_cycles

        # Lưu các thông số  gồm phạt, thưởng
        self.alpha = alpha
        self.collision_reward = collision_reward
        self.beta = beta

        self.env = simple_tag_v3.parallel_env(
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles = max_cycles,
            render_mode = render_mode,
            continuous_actions = True 
        )

        #Lưu tên các agent
        self.agent_order = self.env.possible_agents
        self.predators = [agent for agent in self.agent_order if agent.startswith("adversary_")]
        self.preys = [agent for agent in self.agent_order if agent.startswith("agent_")]

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return observations, infos

    def _get_self_pos(self, obs):
        return obs[2:4]
    
    # Lấy vị trí tương đối của các agent khác so với agent hiện tại
    def get_others_pos(self, agent_name, obs):
        start = 2 + 2 + 2 * self.num_obstacles
        end = start + 2 * (len(self.agent_order) - 1)

        other_pos = obs[start:end].reshape(-1, 2)
        return dict(zip([agent for agent in self.agent_order if agent != agent_name], other_pos))

    def _compute_min_distance_to_preys(self, observations, predator_name):
        pred_obs = observations[predator_name]
        other_pos = self.get_others_pos(predator_name, pred_obs)
        distances = []

        for prey in self.preys:
            if prey in other_pos:
                prey_pos = other_pos[prey]
                distance = np.linalg.norm(prey_pos)
                distances.append(distance)
        
        return min(distances, default=float('inf'))
    
    def _bound_penalty_id(self, value):
        abs_val = abs(value)
        if abs_val < 0.9:
            return 0
        elif 0.9 <= abs_val < 1.0:
            return (abs_val - 0.9) * 10
        elif abs_val >= 1.0:
            return min(1.0 + (abs_val - 1.0) * 20, 10.0)

    def _compute_prey_bound_penalty(self, observations, prey_name):
        prey_obs = observations[prey_name]
        x, y = self._get_self_pos(prey_obs)
        penalty_x = self._bound_penalty_id(x)
        penalty_y = self._bound_penalty_id(y)
        return penalty_x + penalty_y
    
    def step(self, actions):
        next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

        custom_rewards = rewards.copy()

        for predator in self.predators:
            if predator in next_observations:
                min_dis = self._compute_min_distance_to_preys(next_observations, predator)
                custom_rewards[predator]  -= self.alpha * min_dis

        for prey in self.preys:
            if prey in next_observations:
                bound_penalty = self._compute_prey_bound_penalty(next_observations, prey)
                custom_rewards[prey]  -= self.beta * bound_penalty    

        return next_observations, custom_rewards, terminations, truncations, infos 

    def get_global_state(self, observations):
        state = []

        for agent in self.agent_order:
            if(agent in observations):
                state.append(observations[agent])

        return np.concatenate(state).astype(np.float32)

env = CustomPredatorPreyEnv()
obs, info = env.reset()

for step in range(5):
    print(f"\n--- Bước {step} ---")
    actions = {}
    for agent in env.agent_order:
        action = env.env.action_space(agent).sample()  # Lấy hành động ngẫu nhiên
        actions[agent] = action
        print(f"Hành động của {agent}: {action}")

    next_observations, rewards, terminations, truncations, infos = env.step(actions)
    observations = next_observations
    print(f"Phần thưởng: {rewards}")
    print(f"Kết thúc: {terminations}")
    print(f"Truncations: {truncations}")