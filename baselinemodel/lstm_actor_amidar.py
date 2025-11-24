import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # GPU 设置

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

# =====================================================
#  基础设置
# =====================================================

gym.register_envs(ale_py)

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- [修改 1] 游戏 ID 改为 Amidar ---
ENV_ID = "ALE/Amidar-v5"

NUM_ACTORS = 80
UNROLL_LENGTH = 100
BATCH_UNROLLS = 32
MAX_FRAMES = 200_000_000

# --- [修改 2] Gamma 微调 ---
# Amidar 节奏比 Defender 稍快，0.99 或 0.995 比较合适
GAMMA = 0.99
LR = 6e-4      # 稍微降低一点 LR 增加稳定性
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
CLIP_GRAD_NORM = 40.0
FRAME_SKIP = 4

# =====================================================
#  环境封装 (关键修改)
# =====================================================

class FireResetWrapper(gym.Wrapper):
    """
    [修改 3] 修复 FireResetWrapper
    Amidar 必须按 FIRE 才能开始游戏，这里增加了自动检测逻辑
    """
    def __init__(self, env):
        super().__init__(env)
        # 自动检测动作空间里是否有 'FIRE'
        action_meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in action_meanings:
            self.has_fire = True
            # print(f"FireResetWrapper: Enabled for {env.spec.id}")
        else:
            self.has_fire = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.has_fire:
            return obs, info
        
        # 执行 FIRE 动作 (通常动作索引 1 是 FIRE)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        
        # 再执行一步 NOOP (动作 0) 让游戏逻辑走一步
        obs, _, terminated, truncated, _ = self.env.step(0)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            
        return obs, info

def make_atari_env(env_id, seed=None):
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        frame_skip=FRAME_SKIP,
        noop_max=30,
    )
    env = FireResetWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    return env

# =====================================================
#  CNN + LSTM 网络 (保持不变)
# =====================================================

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

class ImpalaCNN_LSTM(nn.Module):
    def __init__(self, input_channels, action_dim, hidden_size=256):
        super().__init__()
        self.block1 = ImpalaBlock(input_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)

        self.fc = nn.Linear(32 * 11 * 11, hidden_size)
        self.relu = nn.ReLU(inplace=True)

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
        )

        self.policy = nn.Linear(hidden_size, action_dim)
        self.value = nn.Linear(hidden_size, 1)

    def get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, x, core_state):
        T, B, C, H, W = x.shape
        x = x.contiguous().view(T * B, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        feat = self.relu(self.fc(x))           
        feat = feat.view(T, B, -1)             

        lstm_out, new_core = self.lstm(feat, core_state)  
        out_flat = lstm_out.contiguous().view(T * B, -1)

        logits = self.policy(out_flat)        
        values = self.value(out_flat).squeeze(-1) 

        return logits, values, new_core

# =====================================================
#  V-trace
# =====================================================

def vtrace(behavior_logp, target_logp, rewards, values, dones,
           gamma=0.99, rho_bar=1.0, c_bar=1.0):
    T, B = rewards.shape

    log_rhos = target_logp - behavior_logp
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    discounts = gamma * (1.0 - dones)

    vs = torch.zeros_like(values)
    vs[-1] = values[-1]

    for t in reversed(range(T)):
        delta = rhos_clipped[t] * (rewards[t] + discounts[t] * values[t+1] - values[t])
        vs[t] = values[t] + delta + discounts[t] * cs[t] * (vs[t+1] - values[t+1])

    vs_t = vs[:-1]
    vs_tp1 = vs[1:]

    pg_adv = rhos_clipped * (rewards + discounts * vs_tp1 - values[:-1])
    return vs_t.detach(), pg_adv.detach()

# =====================================================
#  Actor
# =====================================================

@torch.no_grad()
def select_action(agent, obs, core_state, device):
    obs = np.array(obs, copy=False)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    if obs_t.ndim == 3:
        obs_t = obs_t.unsqueeze(0).unsqueeze(0) 

    logits_flat, _, new_core = agent(obs_t, core_state) 
    logits = logits_flat.view(1, -1) 

    dist = Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)

    return int(action.item()), float(logp.item()), new_core

def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    action_dim = env.action_space.n

    agent = ImpalaCNN_LSTM(obs.shape[0], action_dim).to(device)
    core_state = agent.get_initial_state(batch_size=1, device=device)

    try:
        weights = param_queue.get(timeout=10.0)
        agent.load_state_dict(weights)
    except queue.Empty:
        return

    episode_return = 0.0

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
        episode_returns = []

        init_h = core_state[0].cpu().numpy()
        init_c = core_state[1].cpu().numpy()
        init_h = init_h[:, 0, :] 
        init_c = init_c[:, 0, :] 

        for t in range(unroll_length):
            obs_list.append(np.array(obs, dtype=np.uint8))

            a, logp, core_state = select_action(agent, obs, core_state, device)
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            episode_return += r
            
            # --- [修改 4] 奖励处理：推荐使用 Log-Scaling ---
            # Amidar 也是分数累积型，Clipping 会丢失信息。
            # Log Scaling: sign(r) * log(1 + |r|)
            r_processed = np.sign(r) * np.log(1.0 + np.abs(r))
            
            actions.append(a)
            behavior_logps.append(logp)
            rewards.append(float(r_processed))
            dones.append(1.0 if done else 0.0)

            obs = next_obs
            if done: 
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
                core_state = agent.get_initial_state(1, device)

        obs_list.append(np.array(obs, dtype=np.uint8))

        batch = {
            "obs": np.stack(obs_list, axis=0),              
            "actions": np.array(actions, dtype=np.int64),   
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "logp_b": np.array(behavior_logps, dtype=np.float32),
            "init_h": init_h,                               
            "init_c": init_c,                               
            "real_returns": np.array(episode_returns, dtype=np.float32),
        }

        try:
            data_queue.put(batch)
        except:
            return

# =====================================================
#  Learner
# =====================================================

def learner_train_step(agent, optimizer, batch_list):
    B = len(batch_list)
    T = batch_list[0]["actions"].shape[0]

    obs_batch = np.stack([b["obs"] for b in batch_list], axis=1)       
    actions_batch = np.stack([b["actions"] for b in batch_list], axis=1)
    rewards_batch = np.stack([b["rewards"] for b in batch_list], axis=1)
    dones_batch = np.stack([b["dones"] for b in batch_list], axis=1)
    logp_b_batch = np.stack([b["logp_b"] for b in batch_list], axis=1)

    init_h_batch = np.stack([b["init_h"] for b in batch_list], axis=1) 
    init_c_batch = np.stack([b["init_c"] for b in batch_list], axis=1) 

    obs = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
    actions = torch.tensor(actions_batch, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards_batch, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones_batch, dtype=torch.float32, device=DEVICE)
    logp_b = torch.tensor(logp_b_batch, dtype=torch.float32, device=DEVICE)

    init_h = torch.tensor(init_h_batch, dtype=torch.float32, device=DEVICE)
    init_c = torch.tensor(init_c_batch, dtype=torch.float32, device=DEVICE)
    core_state = (init_h, init_c)

    T1, B1, C, H, W = obs.shape
    assert T1 == T + 1

    logits_flat, values_flat, _ = agent(obs, core_state)
    action_dim = logits_flat.shape[-1]

    logits = logits_flat.view(T1, B1, action_dim) 
    values = values_flat.view(T1, B1)             

    logits_t = logits[:-1]                        
    log_probs_t = F.log_softmax(logits_t, dim=-1)

    actions_expanded = actions.unsqueeze(-1)       
    target_logp = torch.gather(log_probs_t, 2, actions_expanded).squeeze(-1) 

    vs, pg_adv = vtrace(
        behavior_logp=logp_b,
        target_logp=target_logp,
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=GAMMA,
    )

    policy_loss = -(pg_adv * target_logp).mean()
    
    # --- [修改 5] 使用 Huber Loss 提高 Value 学习稳定性 ---
    value_loss = F.huber_loss(values[:-1], vs, delta=1.0)

    probs_t = torch.exp(log_probs_t)
    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

# =====================================================
#  Evaluate
# =====================================================

@torch.no_grad()
def evaluate(env_id, agent, episodes=5):
    env = make_atari_env(env_id)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        core_state = agent.get_initial_state(1, DEVICE)
        while not done:
            a, _, core_state = select_action(agent, obs, core_state, DEVICE)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_r += r
        scores.append(total_r)
    env.close()
    return float(np.mean(scores))

# =====================================================
#  Main
# =====================================================

def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_atari_env(ENV_ID, seed=seed)
    obs, _ = env.reset()
    action_dim = env.action_space.n
    env.close()

    print(f"[IMPALA-Amidar] Env: {ENV_ID}, Obs shape: {obs.shape}, Actions: {action_dim}")
    print(f"[IMPALA-Amidar] Device: {DEVICE}, Actors: {NUM_ACTORS}")
    LR_START = LR
    LR_END = 0.0
    agent = ImpalaCNN_LSTM(obs.shape[0], action_dim).to(DEVICE)
    optimizer = torch.optim.RMSprop(
        agent.parameters(),
        lr=LR_START,
        alpha=0.99,
        eps=0.01,
        momentum=0.0
    )

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    init_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    print("[IMPALA-Amidar] Starting actors...")
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

    records = []
    os.makedirs("checkpoints", exist_ok=True)
    # --- [修改 6] 文件名改为 amidar 防止覆盖 ---
    npy_path = f"checkpoints/bs32_lstm_amidar_records.npy"
    model_path = f"checkpoints/bs32_lstm_amidar.pth"

    try:
        while total_frames < MAX_FRAMES:
            progress = total_frames / MAX_FRAMES
            # 只有当 current_lr 大于 0 时才更新，防止最后变成负数（虽然理论上不会）
            # 线性插值计算当前 LR: 初始值 * (1 - 进度)
            current_lr = LR_START * (1.0 - progress)

            if current_lr < 0: current_lr = 0.0

            # 手动更新优化器里的 LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            batch_list = []
            for _ in range(BATCH_UNROLLS):
                batch = data_queue.get()
                batch_list.append(batch)
                total_frames += UNROLL_LENGTH * FRAME_SKIP

                if "real_returns" in batch and len(batch["real_returns"]) > 0:
                    for ret in batch["real_returns"]:
                        records.append((total_frames, ret))
                        print(f"Episode finished | Frames: {total_frames} | Return: {ret:.2f}")

            loss, pl, vl, ent = learner_train_step(agent, optimizer, batch_list)
            update_count += 1

            if update_count % 5 == 0:
                cpu_state = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try:
                        while not q.empty():
                            q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put(cpu_state)

            if update_count % 100 == 0:
                fps = total_frames / (time.time() - start_time)
                np.save(npy_path, np.array(records, dtype=np.float32))
                print(
                    f"[Upd {update_count:5d}] Frames: {total_frames/1e6:.2f}M | "
                    f"Loss: {loss:.3f} (P {pl:.3f}, V {vl:.3f}, H {ent:.3f}) | FPS: {fps:.0f}"
                    f" | LR: {current_lr:.6f}"
                )

            if update_count % 500 == 0:
                avg_score = evaluate(ENV_ID, agent, episodes=3)
                print(f"=== Eval @ {total_frames/1e6:.2f}M | Avg Score: {avg_score:.2f} ===")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Terminating actors...")
        for p in processes:
            p.terminate()
            p.join()

        torch.save(agent.state_dict(), model_path)
        np.save(npy_path, np.array(records, dtype=np.float32))
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    run()