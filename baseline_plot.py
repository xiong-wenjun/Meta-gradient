import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_baseline(npy_file):
    # 加载 baseline 的结果 (形状应为 30 × 10)
    data = np.load(npy_file)   # shape = (num_seeds, num_points)

    # 计算平均值和标准差
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # x 轴是 iteration（每隔 100 次测试一次）
    iterations = [(i+1) * (1000 // 10) for i in range(data.shape[1])]

    # 绘制
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, mean, label="Baseline Mean", color='blue')
    plt.fill_between(iterations, mean - std, mean + std,
                     color='blue', alpha=0.2, label="Std Range")

    plt.xlabel("Iterations")
    plt.ylabel("Average Return")
    plt.title("Baseline Performance ({} seeds)".format(data.shape[0]))
    plt.grid(True)
    plt.legend()
    plt.savefig("baseline_vtrace_curve.png", dpi=200)
    plt.show()

    print("绘图完成，已保存为 baseline_vtrace_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=False, default="/gemini/code/project/baseline_vtrace_multi_actor_2025_11_21_14_45_50.npy",
                        help="/gemini/code/project/2025_11_21_13_45_01.npy")
    args = parser.parse_args()

    plot_baseline(args.file)