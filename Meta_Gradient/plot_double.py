import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================= 配置区域 =================
# 1. 设置两个文件的路径
path_1 = "/gemini/code/project/baselinemodel/checkpoints/impala_ALE_Defender-v5_records.npy"  # 比如：Baseline
path_2 = "/gemini/code/project/baselinemodel/checkpoints/impala_lstm_ALE_Defender-v5_records.npy"            # 比如：Meta-Gradient

# 2. 设置图例名称 (Legend)
label_1 = "IMPALA (Baseline)"
label_2 = "Meta-Gradient IMPALA"

# 3. 设置颜色
color_1 = "orange"
color_2 = "red" # 或者 'red', 'green' 等

# 4. 平滑窗口大小 (越大越平滑)
SMOOTH_WINDOW = 500 
# ===========================================

def load_and_smooth(path, window_size):
    """
    读取 .npy 文件并计算滑动平均
    """
    try:
        data = np.load(path)
        frames = data[:, 0]
        scores = data[:, 1]
        
        # 使用 Pandas 计算滑动平均
        df = pd.DataFrame({'frames': frames, 'scores': scores})
        smooth_scores = df['scores'].rolling(window=window_size, min_periods=1).mean()
        return df['frames'], smooth_scores
    except FileNotFoundError:
        print(f"错误：找不到文件 {path}，请检查路径。")
        return None, None

# --- 开始绘图 ---
plt.figure(figsize=(12, 6), dpi=150) # 提高 dpi 让图片更清晰

# 1. 处理并绘制第一条曲线
frames_1, scores_1 = load_and_smooth(path_1, SMOOTH_WINDOW)
if frames_1 is not None:
    plt.plot(frames_1, scores_1, color=color_1, linewidth=2, label=label_1)

# 2. 处理并绘制第二条曲线
frames_2, scores_2 = load_and_smooth(path_2, SMOOTH_WINDOW)
if frames_2 is not None:
    plt.plot(frames_2, scores_2, color=color_2, linewidth=2, label=label_2)

# --- 装饰图表 ---
plt.title("Training Curve Comparison: Defender", fontsize=16)
plt.xlabel("Training Frames", fontsize=14)
plt.ylabel("Average Score", fontsize=14)

# 添加网格
plt.grid(True, which='major', linestyle='--', alpha=0.6)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', alpha=0.3)

# 显示图例 (Legend)
plt.legend(fontsize=12, loc='upper left')

# 保存图片
save_filename = "/gemini/code/defender_comparison_curve.png"
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
print(f"对比图已保存为: {save_filename}")

# 显示
plt.show()