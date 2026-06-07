import torch
from mpe2 import simple_tag_v3
import time
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# Import class MADDPGAgent từ file agent.py của bạn
from maddpg.agent import MADDPGAgent

def evaluate(episodes=5):
    # 1. Khởi tạo môi trường có Đồ Họa (render_mode="human")
    env = simple_tag_v3.parallel_env(render_mode="human", continuous_actions=False)
    env.reset()
    
    # Evaluate thì chỉ cần chạy trên CPU là đủ nhanh và không tốn VRAM
    device = torch.device("cpu") 
    
    # 2. Khởi tạo danh sách Agents
    agents = []
    for i in range(env.num_agents):
        # Khởi tạo agent giống hệt lúc train
        agent = MADDPGAgent(agent_id=i, device=device)
        
        # LOAD TRỌNG SỐ TỪ CHECKPOINT CỦA BẠN (Sửa lại tên file cho đúng nếu cần)
        try:
            agent.actor.load_state_dict(torch.load(f"actor_agent_{i}.pth", map_location=device))
            print(f"Đã load thành công não cho Agent {i}")
        except FileNotFoundError:
            print(f"Không tìm thấy file actor_agent_{i}.pth! Hãy đảm bảo bạn đã chạy train.py và lưu model.")
            return
            
        # Chuyển mạng sang chế độ đánh giá (tắt Dropout/BatchNorm nếu có)
        agent.actor.eval()
        agents.append(agent)

    print("\n--- BẮT ĐẦU SHOWTIME ---")
    print("Mẹo: Nhấn Ctrl + C ở cửa sổ Terminal (dòng lệnh) để tắt ngay lập tức.")
    # 3. Vòng lặp quan sát
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(25): # Mặc định MPE simple_tag có 25 steps / episode
                import numpy as np
                for a in env.agents:
                    if len(obs[a]) < 16:
                        obs[a] = np.pad(obs[a], (0, 16 - len(obs[a])), 'constant')

                actions = {}
                
                for agent_id, agent_name in enumerate(env.possible_agents):
                    if agent_name in env.agents:
                        # RẤT QUAN TRỌNG: explore=False để bỏ Gumbel noise, chỉ dùng argmax lấy action tốt nhất
                        action = agents[agent_id].select_action(obs[agent_name], explore=False)
                        actions[agent_name] = action
                    
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # Làm chậm nhịp độ một chút để mắt người kịp nhìn
                # Nếu muốn nhanh hơn, bạn có thể xóa hẳn dòng time.sleep đi
                time.sleep(0.01) 
                
                obs = next_obs
                episode_reward += sum(rewards.values())
                
                if all(terminations.values()) or all(truncations.values()):
                    break
                    
            print(f"Hiệp {ep + 1} kết thúc. Tổng Reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\nĐã ép buộc dừng xem sớm bằng Ctrl + C!")
        
    finally:
        env.close()

if __name__ == "__main__":
    evaluate()