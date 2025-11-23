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

# 注册 ALE 游戏
gym.register_envs(ale_py)

# 多进程启动方式
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------------------
#  设备与全局超参数 (针对 Defender 调整)
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ====== 修改 1: 游戏改为 Defender ======
ENV_ID = "ALE/Defender-v5" 

# ====== 修改 2: 保持高并行度 ======
NUM_ACTORS = 80                 # Defender 需要大量样本来学习躲避和射击
UNROLL_LENGTH = 20              
BATCH_UNROLLS = 32              
MAX_FRAMES = 20_000_000         # 建议保持 20M 或更多
GAMMA = 0.99

# ====== 修改 3: 学习率 ======
# Defender 的奖励反馈比 Tennis 稍微密集一点（射击敌人得分），
# 但 5e-4 依然是一个非常稳健的起点。
LR = 6e-4                       
ENTROPY_COEF = 0.01             
VALUE_COEF = 0.5
CLIP_GRAD_NORM = 40.0            

FRAME_SKIP = 4                  

# ---------------------------
#  辅助 Wrapper: 自动开始游戏 (Fire Reset)
# ---------------------------
class FireResetWrapper(gym.Wrapper):
    """
    Defender 在 reset 后也需要执行 'FIRE' 动作才能从标题画面开始游戏。
    """
    def __init__(self, env):
        super().__init__(env)
        # 检查动作 1 是否为 FIRE (Atari 标准)
        # Defender 的动作 1 通常也是 FIRE
        if len(env.unwrapped.get_action_meanings()) >= 3:
             assert env.unwrapped.get_action_meanings()[1] == 'FIRE'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 执行一次 FIRE 动作 (Action Index 1) 尝试开始游戏
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        # 再执行一次 NOOP 缓冲
        obs, _, terminated, truncated, _ = self.env.step(0)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

# ---------------------------
#  环境构造函数
# ---------------------------
def make_atari_env(env_id, seed=None):
    """
    构造 Atari 环境
    """
    env = gym.make(env_id, frameskip=1)

    # Atari 预处理
    env = gym.wrappers.AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False, 
        frame_skip=FRAME_SKIP,
        noop_max=30      
    )

    # 挂载 FireResetWrapper (对 Defender 也很重要)
    env = FireResetWrapper(env)

    # Frame Stack
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    if seed is not None:
        env.reset(seed=seed)

    return env

# ---------------------------
#  IMPALA CNN 网络结构 (ResNet)
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
        
        self.fc = nn.Linear(32 * 11 * 11, 256)
        
        # ====== 自动适应动作空间 ======
        # Defender 会传入 action_dim=18
        self.policy = nn.Linear(256, action_dim) 
        self.value = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc(x))
        logits = self.policy(x)            
        value = self.value(x).squeeze(-1)  
        return logits, value

# ---------------------------
#  V-Trace 计算
# ---------------------------
def vtrace(behavior_logp, target_logp, rewards, values, dones, gamma=0.99, rho_bar=1.0, c_bar=1.0):
    log_rhos = target_logp - behavior_logp       
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    discounts = gamma * (1.0 - dones)           

    T, B = rewards.shape
    vs = torch.zeros_like(values)               
    vs[-1] = values[-1]                         

    for t in reversed(range(T)):
        delta = rhos_clipped[t] * (
            rewards[t] + discounts[t] * values[t + 1] - values[t]
        )
        vs[t] = values[t] + delta + discounts[t] * cs[t] * (vs[t + 1] - values[t + 1])

    vs_t = vs[:-1]          
    vs_tp1 = vs[1:]         
    pg_adv = rhos_clipped * (rewards + discounts * vs_tp1 - values[:-1])

    return vs_t.detach(), pg_adv.detach()

# ---------------------------
#  选动作函数
# ---------------------------
@torch.no_grad()
def select_action(agent, obs, device):
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
#  Actor 进程
# ---------------------------
def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()

    obs_shape = np.array(obs).shape
    action_dim = env.action_space.n  # Defender 这里会自动变成 18

    agent = ImpalaCNN(obs_shape[0], action_dim).to(device)

    try:
        weights = param_queue.get(timeout=10.0)
        agent.load_state_dict(weights)
    except queue.Empty:
        return

    while True:
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

        obs_list.append(np.array(obs, dtype=np.uint8))

        batch = {
            "obs": np.stack(obs_list, axis=0),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "logp_b": np.array(behavior_logps, dtype=np.float32),
        }
        data_queue.put(batch)

# ---------------------------
#  Learner 训练步
# ---------------------------
def learner_train_step(agent, optimizer, batch_list):
    B = len(batch_list)
    T = batch_list[0]["actions"].shape[0]

    obs_batch = np.stack([b["obs"] for b in batch_list], axis=1)         
    actions_batch = np.stack([b["actions"] for b in batch_list], axis=1) 
    rewards_batch = np.stack([b["rewards"] for b in batch_list], axis=1) 
    dones_batch = np.stack([b["dones"] for b in batch_list], axis=1)     
    logp_b_batch = np.stack([b["logp_b"] for b in batch_list], axis=1)   

    obs = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
    actions = torch.tensor(actions_batch, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards_batch, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones_batch, dtype=torch.float32, device=DEVICE)
    logp_b = torch.tensor(logp_b_batch, dtype=torch.float32, device=DEVICE)

    T1, B1, C, H, W = obs.shape
    obs_flat = obs.view(T1 * B1, C, H, W)
    logits_flat, values_flat = agent(obs_flat)

    action_dim = logits_flat.shape[-1]
    logits = logits_flat.view(T1, B1, action_dim)
    values = values_flat.view(T1, B1)

    logits_t = logits[:-1]
    values_t = values
    log_probs_t = F.log_softmax(logits_t, dim=-1)

    actions_expanded = actions.unsqueeze(-1)
    logp_t = torch.gather(log_probs_t, dim=2, index=actions_expanded).squeeze(-1)

    vs, pg_adv = vtrace(
        behavior_logp=logp_b,
        target_logp=logp_t,
        rewards=rewards,
        values=values_t,
        dones=dones,
        gamma=GAMMA,
        rho_bar=1.0,
        c_bar=1.0,
    )

    policy_loss = -(pg_adv * logp_t).mean()
    v_pred = values_t[:-1]
    value_loss = F.mse_loss(v_pred, vs)

    probs_t = torch.exp(log_probs_t)
    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

# ---------------------------
#  评估函数
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
#  主函数
# ---------------------------
def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 探测环境，Defender action_dim 自动为 18
    env = make_atari_env(ENV_ID, seed=seed)
    obs, _ = env.reset()
    obs_shape = np.array(obs).shape
    action_dim = env.action_space.n 
    env.close()

    print(f"[IMPALA] Env: {ENV_ID}, Obs shape: {obs_shape}, Actions: {action_dim}")
    print(f"[IMPALA] Device: {DEVICE}, Actors: {NUM_ACTORS}")

    agent = ImpalaCNN(obs_shape[0], action_dim).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

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

    os.makedirs("checkpoints", exist_ok=True)

    npy_path = f"checkpoints/impala_{ENV_ID.replace('/', '_')}_records.npy"
    model_path = f"checkpoints/impala_{ENV_ID.replace('/', '_')}.pth"

    records = []

    try:
        while total_frames < MAX_FRAMES:
            batch_list = []
            for _ in range(BATCH_UNROLLS):
                batch = data_queue.get()
                batch_list.append(batch)
                total_frames += UNROLL_LENGTH * FRAME_SKIP

            loss, pl, vl, ent = learner_train_step(agent, optimizer, batch_list)
            update_count += 1

            if update_count % 10 == 0: 
                state_dict_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try:
                        while not q.empty():
                            q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put(state_dict_cpu)

            if update_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = total_frames / elapsed
                print(
                    f"[Update {update_count:5d}] Frames: {total_frames/1e6:.2f}M | "
                    f"Loss: {loss:.3f} (P {pl:.3f}, V {vl:.3f}, H {ent:.3f}) | FPS: {fps:.0f}"
                )

            if update_count % 500 == 0:
                avg_score = evaluate(ENV_ID, agent, episodes=3)
                records.append((total_frames, avg_score))
                print(
                    f"=== Eval @ {total_frames/1e6:.2f}M frames | "
                    f"Avg Score: {avg_score:.2f} | Time: {time.time() - start_time:.1f}s ==="
                )
                np.save(npy_path, np.array(records))
                print(f"Running data saved to {npy_path}")


    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Terminating actors...")
        for p in processes:
            p.terminate()
            p.join()

    
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 保存最终的曲线数据
    np.save(npy_path, np.array(records))
    print(f"Final training curve saved to {npy_path}")

if __name__ == "__main__":
    run(seed=42)