import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 讀取數據 (請確保路徑正確)
path = "/gemini/code/project/baselinemodel/checkpoints/impala_ALE_Defender-v5_records.npy"
try:
    data = np.load(path)
except FileNotFoundError:
    print(f"錯誤：找不到文件 {path}，請檢查路徑。")
    exit()

frames = data[:, 0]
scores = data[:, 1]

# 2. 使用 Pandas 進行滑動平均 (Smoothing)
# window=100 表示計算最近 100 局遊戲的平均分，數值越大曲線越平滑
df = pd.DataFrame({'frames': frames, 'scores': scores})
df['smooth'] = df['scores'].rolling(window=100, min_periods=1).mean()

# 3. 設置畫布大小
plt.figure(figsize=(12, 6), dpi=100) # dpi=100 讓顯示更清晰

# 畫原始數據 (淺色，作為背景噪聲)
#plt.plot(df['frames'], df['scores'], color='lightblue', alpha=0.3, label='Raw Data (Real-time)')

# 畫平滑數據 (深色，作為主趨勢線)
plt.plot(df['frames'], df['smooth'], color='blue', linewidth=2, label='Smoothed (Moving Avg)')

# 設置標題和標籤
plt.title("IMPALA (ResNet) Training Curve - Defender", fontsize=14)
plt.xlabel("Frames", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

# === 核心修改：保存圖片 ===
# dpi=300 表示保存為高分辨率圖片，適合放進論文或報告
save_filename = "/gemini/code/defender_training_curve.png"
plt.savefig(save_filename, dpi=300, bbox_inches='tight')
print(f"圖片已保存為: {save_filename}")

# 顯示圖片
plt.show()
