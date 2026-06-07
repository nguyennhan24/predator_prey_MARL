import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    """Làm mượt đồ thị bằng trung bình trượt"""
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def plot_learning_curve(rewards_file="rewards.npy"):
    """
    Giả định trong file train.py của bạn có lưu lại mảng Reward:
    np.save("rewards.npy", danh_sach_reward_sau_khi_train)
    """
    try:
        # Load file numpy chứa dữ liệu log
        rewards = np.load(rewards_file)
    except FileNotFoundError:
        # TẠO DỮ LIỆU GIẢ ĐỂ BẠN XEM TRƯỚC FORM BIỂU ĐỒ 
        print("Chưa có file log thực tế, tạo dữ liệu sample để demo...")
        episodes = np.arange(1000)
        # Mô phỏng Reward tăng dần theo thời gian (có nhiễu)
        rewards = -50 + 40 * (1 - np.exp(-episodes/300)) + np.random.normal(0, 5, 1000)

    # Tính toán đường mượt
    window = 50
    smoothed_rewards = moving_average(rewards, window_size=window)

    # Thiết lập matplotlib
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Vẽ 2 đường: Dữ liệu gốc (mờ) và Đường trung bình (đậm)
    plt.plot(rewards, alpha=0.3, color='royalblue', label='Reward gốc / Hiệp')
    
    # Căn chỉnh trục X cho đường smoothed (bắt đầu từ vị trí window)
    x_smoothed = np.arange(window-1, len(rewards))
    plt.plot(x_smoothed, smoothed_rewards, color='firebrick', linewidth=2, label=f'Trung bình trượt ({window} hiệp)')

    # Trang trí
    plt.title('MADDPG Learning Curve (Simple Tag)', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Lưu và hiển thị
    plt.savefig('maddpg_learning_curve.png', dpi=300)
    print("Đã lưu biểu đồ thành maddpg_learning_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()