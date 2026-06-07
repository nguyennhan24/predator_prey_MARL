import traceback
import numpy as np
from mpe2 import simple_tag_v3

try:
    env = simple_tag_v3.parallel_env(render_mode=None, continuous_actions=False, max_cycles=25)
    obs, info = env.reset()
    for i in range(26):
        actions = {a: 0 for a in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        print(f"step {i}: env.agents:", env.agents)
        print(f"step {i}: next_obs keys:", next_obs.keys())
        if not env.agents:
            break
except Exception as e:
    traceback.print_exc()
