import traceback
from mpe2 import simple_tag_v3

try:
    env = simple_tag_v3.parallel_env(render_mode=None, continuous_actions=False)
    obs, info = env.reset()
    print("Initial env.agents:", env.agents)
    print("obs keys:", obs.keys())
    for k, v in obs.items():
        print(f"obs[{k}] shape: {v.shape}")
except Exception as e:
    traceback.print_exc()
