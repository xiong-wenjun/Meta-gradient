import os
import time
import random
import queue
import numpy as np

import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

# 注册 ALE 游戏到 gymnasium
gym.register_envs(ale_py)

# 多进程启动方式
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------------------
#  设备与全局超参数
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ====== 这里改游戏 ======
ENV_ID = "ALE/Pong-v5"          # 先用 Pong 验证，之后可以改成 ALE/Tennis-v5 等

# IMPALA 相关超参
NUM_ACTORS = 8                  # actor 进程数，多核可以适当调高
UNROLL_LENGTH = 20              # 每个 actor 每次 rollout 的时间步 T
BATCH_UNROLLS = 16              # learner 每次更新使用多少个 unroll
MAX_FRAMES = 10_000_000         # 总环境帧数上限（10M 够你先试）
GAMMA = 0.99

LR = 1e-3
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
CLIP_GRAD_NORM = 5.0

FRAME_SKIP = 4                  # AtariPreprocessing 里的 frame_skip 要对应


# ---------------------------
#   环境构造函数
# ---------------------------
def make_atari_env(env_id, seed=None):
    """
    构造 Atari 环境：84x84 灰度 + frame_skip + 帧堆叠4帧
    """
    env = gym.make(env_id, frameskip=1)  # 让 AtariPreprocessing 控制跳帧

    env = gym.wrappers.AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,  # obs 为 uint8，之后 /255.0 归一化
        frame_skip=FRAME_SKIP
    )

    # 你原来的代码用的是 FrameStackObservation，就沿用
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    if seed is not None:
        env.reset(seed=seed)

    return env


# ---------------------------
#   IMPALA CNN 网络结构
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super().__init__()
        self.block1 = ImpalaBlock(input_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)

        # 84x84 经过 3 次 pool 后 roughly 11x11
        self.fc = nn.Linear(32 * 11 * 11, 256)
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B,C,H,W]，float in [0,1]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc(x))
        logits = self.policy(x)            # [B, A]
        value = self.value(x).squeeze(-1)  # [B]
        return logits, value


# ---------------------------
#   V-Trace 计算（time-major）
# ---------------------------
def vtrace(
    behavior_logp,  # [T,B]
    target_logp,    # [T,B]
    rewards,        # [T,B]
    values,         # [T+1,B]
    dones,          # [T,B]  1.0 if done else 0.0
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
):
    """
    Espeholt et al. 2018 中的 V-trace 算法 (batched, time-major).

    values: V(s_0..s_T) -> [T+1,B]
    behavior_logp: log μ(a_t | s_t)
    target_logp:   log π(a_t | s_t)
    """
    log_rhos = target_logp - behavior_logp       # [T,B]
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    discounts = gamma * (1.0 - dones)           # [T,B]

    T, B = rewards.shape
    vs = torch.zeros_like(values)               # [T+1,B]
    vs[-1] = values[-1]                         # bootstrap

    # 反向递推 V-trace 目标
    for t in reversed(range(T)):
        delta = rhos_clipped[t] * (
            rewards[t] + discounts[t] * values[t + 1] - values[t]
        )
        vs[t] = values[t] + delta + discounts[t] * cs[t] * (vs[t + 1] - values[t + 1])

    # 用于 policy gradient 的优势
    vs_t = vs[:-1]          # [T,B]
    vs_tp1 = vs[1:]         # [T,B]
    pg_adv = rhos_clipped * (rewards + discounts * vs_tp1 - values[:-1])

    return vs_t.detach(), pg_adv.detach()


# ---------------------------
#   选动作函数（Actor 用）
# ---------------------------
@torch.no_grad()
def select_action(agent, obs, device):
    """
    obs: LazyFrames / np.uint8, shape [C,H,W]
    """
    obs = np.array(obs, copy=False)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    if obs_t.ndim == 3:
        obs_t = obs_t.unsqueeze(0)
    logits, value = agent(obs_t)
    probs = F.softmax(logits, dim=-1)
    dist = Categorical(probs)
    action = dist.sample()
    logp = dist.log_prob(action)
    return int(action.item()), float(logp.item())


# ---------------------------
#   Actor 进程：只负责采样
# ---------------------------
def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()

    obs_shape = np.array(obs).shape
    action_dim = env.action_space.n

    agent = ImpalaCNN(obs_shape[0], action_dim).to(device)

    # 初次同步参数
    try:
        weights = param_queue.get(timeout=10.0)
        agent.load_state_dict(weights)
    except queue.Empty:
        print(f"[Actor {rank}] initial weight sync failed.")
        return

    while True:
        # 尝试获取最新参数（非阻塞）
        try:
            while True:
                new_w = param_queue.get_nowait()
                weights = new_w
        except queue.Empty:
            pass
        agent.load_state_dict(weights)

        obs_list = []
        actions = []
        rewards = []
        dones = []
        behavior_logps = []

        for t in range(unroll_length):
            obs_list.append(np.array(obs, dtype=np.uint8))

            a, logp = select_action(agent, obs, device)
            actions.append(a)
            behavior_logps.append(logp)

            obs_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            r = float(np.clip(r, -1, 1))

            rewards.append(r)
            dones.append(1.0 if done else 0.0)

            obs = obs_next
            if done:
                obs, _ = env.reset()

        # 加上最后一个 s_T
        obs_list.append(np.array(obs, dtype=np.uint8))

        batch = {
            "obs": np.stack(obs_list, axis=0),               # [T+1,C,H,W]
            "actions": np.array(actions, dtype=np.int64),   # [T]
            "rewards": np.array(rewards, dtype=np.float32), # [T]
            "dones": np.array(dones, dtype=np.float32),     # [T]
            "logp_b": np.array(behavior_logps, dtype=np.float32),  # [T]
        }
        data_queue.put(batch)


# ---------------------------
#   Learner：单步训练
# ---------------------------
def learner_train_step(agent, optimizer, batch_list):
    """
    batch_list: 长度 BATCH_UNROLLS 的列表，每个元素为：
        obs: [T+1,C,H,W]
        actions: [T]
        rewards: [T]
        dones: [T]
        logp_b: [T]
    """
    B = len(batch_list)
    T = batch_list[0]["actions"].shape[0]

    # 按 batch 维度把多个 unroll 拼起来，得到 time-major 的数据
    obs_batch = np.stack([b["obs"] for b in batch_list], axis=1)         # [T+1,B,C,H,W]
    actions_batch = np.stack([b["actions"] for b in batch_list], axis=1) # [T,B]
    rewards_batch = np.stack([b["rewards"] for b in batch_list], axis=1) # [T,B]
    dones_batch = np.stack([b["dones"] for b in batch_list], axis=1)     # [T,B]
    logp_b_batch = np.stack([b["logp_b"] for b in batch_list], axis=1)   # [T,B]

    obs = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
    actions = torch.tensor(actions_batch, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards_batch, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones_batch, dtype=torch.float32, device=DEVICE)
    logp_b = torch.tensor(logp_b_batch, dtype=torch.float32, device=DEVICE)

    T1, B1, C, H, W = obs.shape
    obs_flat = obs.view(T1 * B1, C, H, W)
    logits_flat, values_flat = agent(obs_flat)  # logits: [T1*B,A], values: [T1*B]

    action_dim = logits_flat.shape[-1]
    logits = logits_flat.view(T1, B1, action_dim)  # [T+1,B,A]
    values = values_flat.view(T1, B1)              # [T+1,B]

    # 取 t=0..T-1 的 logits 对应动作
    logits_t = logits[:-1]                         # [T,B,A]
    values_t = values                              # [T+1,B]

    log_probs_t = F.log_softmax(logits_t, dim=-1)  # [T,B,A]

    actions_expanded = actions.unsqueeze(-1)       # [T,B,1]
    logp_t = torch.gather(log_probs_t, dim=2, index=actions_expanded).squeeze(-1)  # [T,B]

    # V-trace
    vs, pg_adv = vtrace(
        behavior_logp=logp_b,
        target_logp=logp_t,
        rewards=rewards,
        values=values_t,
        dones=dones,
        gamma=GAMMA,
        rho_bar=1.0,
        c_bar=1.0,
    )  # vs: [T,B], pg_adv: [T,B]

    # Policy loss
    policy_loss = -(pg_adv * logp_t).mean()

    # Value loss
    v_pred = values_t[:-1]
    value_loss = F.mse_loss(v_pred, vs)

    # Entropy
    probs_t = torch.exp(log_probs_t)
    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()


# ---------------------------
#   测试 Agent 的平均得分
# ---------------------------
@torch.no_grad()
def evaluate(env_id, agent, episodes=5):
    env = make_atari_env(env_id)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE) / 255.0
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)
            logits, _ = agent(obs_t)
            a = torch.argmax(logits, dim=-1).item()
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_r += r
        scores.append(total_r)
    return float(np.mean(scores))


# ---------------------------
#   主训练循环（Learner）
# ---------------------------
def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 先用一个 env 探测 obs shape 和 action 数
    env = make_atari_env(ENV_ID, seed=seed)
    obs, _ = env.reset()
    obs_shape = np.array(obs).shape
    action_dim = env.action_space.n
    env.close()

    print(f"[IMPALA] Env: {ENV_ID}, Obs shape: {obs_shape}, Actions: {action_dim}")
    print(f"[IMPALA] Device: {DEVICE}, Actors: {NUM_ACTORS}")

    agent = ImpalaCNN(obs_shape[0], action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    # 启动 actor
    processes = []
    init_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    print("[IMPALA] Starting actors...")
    for rank in range(NUM_ACTORS):
        param_queues[rank].put(init_state_dict)
        p = mp.Process(
            target=actor_process,
            args=(rank, ENV_ID, UNROLL_LENGTH, data_queue, param_queues[rank], seed + 1000 + rank),
        )
        p.daemon = True
        p.start()
        processes.append(p)

    total_frames = 0
    update_count = 0
    start_time = time.time()

    records = []  # [(frames, score), ...]

    try:
        while total_frames < MAX_FRAMES:
            # 收集多个 unroll，组成一个大 batch
            batch_list = []
            for _ in range(BATCH_UNROLLS):
                batch = data_queue.get()
                batch_list.append(batch)
                total_frames += UNROLL_LENGTH * FRAME_SKIP

            loss, pl, vl, ent = learner_train_step(agent, optimizer, batch_list)
            update_count += 1

            # 定期同步参数给 actors
            if update_count % 20 == 0:
                state_dict_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try:
                        while not q.empty():
                            q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put(state_dict_cpu)

            # 训练日志
            if update_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = total_frames / elapsed
                print(
                    f"[Update {update_count:5d}] Frames: {total_frames/1e6:.2f}M | "
                    f"Loss: {loss:.3f} (P {pl:.3f}, V {vl:.3f}, H {ent:.3f}) | FPS: {fps:.0f}"
                )

            # 定期评估
            if update_count % 200 == 0:
                avg_score = evaluate(ENV_ID, agent, episodes=5)
                records.append((total_frames, avg_score))
                print(
                    f"=== Eval @ {total_frames/1e6:.2f}M frames | "
                    f"Avg Score: {avg_score:.2f} | Time: {time.time() - start_time:.1f}s ==="
                )

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Terminating actors...")
        for p in processes:
            p.terminate()
            p.join()

    # 保存模型与曲线数据
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/impala_{ENV_ID.replace('/', '_')}.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    npy_path = f"impala_{ENV_ID.replace('/', '_')}_records.npy"
    np.save(npy_path, np.array(records, dtype=np.float32))
    print(f"Training curve saved to {npy_path}")

    return records


if __name__ == "__main__":
    run(seed=42)