BUFFER_SIZE = int(1e6)
BATCH_SIZE = 1024       # Kích thước batch lấy ra mỗi lần học
GAMMA = 0.95            # Hệ số chiết khấu (Discount factor) cho phần thưởng
TAU = 0.01              # Hệ số cập nhật mềm (Soft update) cho Target Networks
LR_ACTOR = 1e-2         # Tốc độ học (Learning rate) của Actor
LR_CRITIC = 1e-2        # Tốc độ học (Learning rate) của Critic

AGENT_ORDER = [
    'adversary_0', 
    'adversary_1', 
    'adversary_2', 
    'agent_0'
]

NUM_AGENTS = len(AGENT_ORDER)

MAX_OBS_DIM = 0

GLOBAL_STATE_DIM = 0