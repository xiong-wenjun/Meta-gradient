import numpy as np
import matplotlib.pyplot as plt

# === 修改你的文件名 ===
file_path = "/gemini/code/impala_ALE_Pong-v5_records.npy"

# 读取
records = np.load(file_path, allow_pickle=True)
records = np.array(records, dtype=np.float32)

frames = records[:, 0]
scores = records[:, 1]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(frames, scores, label="Score", color="blue")

plt.xlabel("Frames")
plt.ylabel("Score")
plt.title("IMPALA Training Curve (Pong)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === 自动保存图像 ===
output_path = "/gemini/code/project/training_curve.png"
plt.savefig(output_path, dpi=200)

print(f"曲线已保存为: {output_path}")

# 显示图像
plt.show()
