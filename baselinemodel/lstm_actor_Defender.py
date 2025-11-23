import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------------------
#  Global Hyperparameters
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True # A100 speedup
    torch.backends.cudnn.allow_tf32 = True

ENV_ID = "ALE/Defender-v5"
NUM_ACTORS = 80             
UNROLL_LENGTH = 20          
BATCH_UNROLLS = 32          
MAX_FRAMES = 200_000_000    
GAMMA = 0.99
LR = 6e-4                   
ENTROPY_COEF = 0.01         
VALUE_COEF = 0.5
CLIP_GRAD_NORM = 40.0       
FRAME_SKIP = 4              

# ---------------------------
#  Wrappers
# ---------------------------
class FireResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(env.unwrapped.get_action_meanings()) >= 3:
             assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, t, tr, _ = self.env.step(1)
        if t or tr: self.env.reset(**kwargs)
        obs, _, t, tr, _ = self.env.step(0)
        if t or tr: self.env.reset(**kwargs)
        return obs, info

def make_atari_env(env_id, seed=None):
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, scale_obs=False, 
        frame_skip=FRAME_SKIP, noop_max=30
    )
    env = FireResetWrapper(env)
    # LSTM 不需要 FrameStack，因为它有记忆！
    # 如果显存够，保留 FrameStack 也可以，但通常 LSTM 配合单帧或少帧即可。
    # 这里为了保持输入一致性，我们依然保留 FrameStack(4)，这叫 "Stacked LSTM"
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    if seed is not None: env.reset(seed=seed)
    return env

# ---------------------------
#  ResNet + LSTM Model
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

class ImpalaCNN_LSTM(nn.Module):
    def __init__(self, input_channels, action_dim):
        super().__init__()
        self.block1 = ImpalaBlock(input_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        
        self.fc = nn.Linear(32 * 11 * 11, 256)
        self.relu = nn.ReLU(inplace=True)
        
        # === 新增 LSTM 层 ===
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=False)
        
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x, core_state):
        """
        x: [T, B, C, H, W]  (Time, Batch, ...)
        core_state: (h, c) tuple, each is [1, B, 256]
        """
        T, B, C, H, W = x.shape
        
        # 1. CNN 提取特征 (合并 T*B 并行处理)
        x = x.view(T * B, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        feat = self.relu(self.fc(x))
        
        # 2. 还原序列形状给 LSTM
        feat = feat.view(T, B, -1)
        
        # 3. LSTM 处理
        lstm_out, (h_n, c_n) = self.lstm(feat, core_state)
        
        # 4. 输出头 (合并 T*B)
        lstm_out_flat = lstm_out.view(T * B, -1)
        logits = self.policy(lstm_out_flat)
        value = self.value(lstm_out_flat).squeeze(-1)
        
        return logits, value, (h_n, c_n)

    # 方便 Actor 单步使用的 helper
    def get_initial_state(self, batch_size, device):
        # LSTM 初始状态: h_0, c_0 都是 0
        return (
            torch.zeros(1, batch_size, 256, device=device),
            torch.zeros(1, batch_size, 256, device=device)
        )

# ---------------------------
#  V-Trace
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
        delta = rhos_clipped[t] * (rewards[t] + discounts[t] * values[t+1] - values[t])
        vs[t] = values[t] + delta + discounts[t] * cs[t] * (vs[t+1] - values[t+1])
        
    vs_t = vs[:-1]
    vs_tp1 = vs[1:]
    pg_adv = rhos_clipped * (rewards + discounts * vs_tp1 - values[:-1])
    return vs_t.detach(), pg_adv.detach()

# ---------------------------
#  Actor Process (With Memory)
# ---------------------------
@torch.no_grad()
def select_action(agent, obs, core_state, device):
    # obs: [C, H, W] -> [1, 1, C, H, W] (Time=1, Batch=1)
    obs = np.array(obs, copy=False)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    if obs_t.ndim == 3: obs_t = obs_t.unsqueeze(0).unsqueeze(0)
    
    logits, _, new_core_state = agent(obs_t, core_state)
    
    logits = logits.squeeze(0) # [1, A]
    dist = Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    
    return int(action.item()), float(logp.item()), new_core_state

def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    
    agent = ImpalaCNN_LSTM(obs.shape[0], env.action_space.n).to(device)
    
    # 初始化 LSTM 状态
    core_state = agent.get_initial_state(batch_size=1, device=device)
    
    try: weights = param_queue.get(timeout=10.0)
    except queue.Empty: return
    agent.load_state_dict(weights)
    
    current_episode_score = 0.0

    while True:
        try: 
            while True: weights = param_queue.get_nowait()
        except queue.Empty: pass
        agent.load_state_dict(weights)
        
        obs_l, act_l, rew_l, don_l, log_l = [], [], [], [], []
        completed_episode_scores = []
        
        # === 关键：记录本段 Unroll 开始时的 LSTM 状态 ===
        # 这样 Learner 才能从这个状态接着往下算
        initial_h = core_state[0].squeeze(1).numpy() # [1, 256]
        initial_c = core_state[1].squeeze(1).numpy() # [1, 256]

        for _ in range(unroll_length):
            obs_l.append(np.array(obs, dtype=np.uint8))
            
            # 传入 core_state, 接收 new_core_state
            a, logp, core_state = select_action(agent, obs, core_state, device)
            
            obs_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            current_episode_score += r
            r = float(np.clip(r, -1, 1))
            
            act_l.append(a); log_l.append(logp); rew_l.append(r); don_l.append(1.0 if done else 0.0)
            
            obs = obs_next
            if done: 
                completed_episode_scores.append(current_episode_score)
                current_episode_score = 0.0
                obs, _ = env.reset()
                # 游戏结束，重置记忆！
                core_state = agent.get_initial_state(1, device)

        obs_l.append(np.array(obs, dtype=np.uint8))
        
        batch = {
            "obs": np.stack(obs_l, axis=0),
            "actions": np.array(act_l),
            "rewards": np.array(rew_l),
            "dones": np.array(don_l),
            "logp_b": np.array(log_l),
            "real_scores": np.array(completed_episode_scores, dtype=np.float32),
            # 附带 LSTM 初始状态
            "init_h": initial_h, 
            "init_c": initial_c
        }
        try: data_queue.put(batch)
        except: return

# ---------------------------
#  Learner Training
# ---------------------------
def learner_train_step(agent, optimizer, batch_list):
    # batch_list 中每个 batch 包含 "init_h" 和 "init_c"
    
    # 1. 堆叠普通数据
    obs_batch = np.stack([b["obs"] for b in batch_list], axis=1) # [T+1, B, C, H, W]
    actions_batch = np.stack([b["actions"] for b in batch_list], axis=1)
    rewards_batch = np.stack([b["rewards"] for b in batch_list], axis=1)
    dones_batch = np.stack([b["dones"] for b in batch_list], axis=1)
    logp_b_batch = np.stack([b["logp_b"] for b in batch_list], axis=1)
    
    # 2. 堆叠 LSTM 初始状态
    # init_h: list of [1, 256] -> stack -> [1, B, 256]
    init_h_batch = np.stack([b["init_h"] for b in batch_list], axis=1) 
    init_c_batch = np.stack([b["init_c"] for b in batch_list], axis=1)
    
    # 转 Tensor
    obs = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
    actions = torch.tensor(actions_batch, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(rewards_batch, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones_batch, dtype=torch.float32, device=DEVICE)
    logp_b = torch.tensor(logp_b_batch, dtype=torch.float32, device=DEVICE)
    
    init_h = torch.tensor(init_h_batch, dtype=torch.float32, device=DEVICE)
    init_c = torch.tensor(init_c_batch, dtype=torch.float32, device=DEVICE)
    core_state = (init_h, init_c) # Learner 开始推演的起点

    # 3. 前向传播 (带状态)
    T1, B1, C, H, W = obs.shape
    # agent forward 需要 [T, B, ...] 格式
    # 这里 obs 包含 T+1 帧，我们一次性传进去，LSTM 会处理序列
    logits_flat, values_flat, _ = agent(obs, core_state)
    
    # 还原形状
    action_dim = logits_flat.shape[-1]
    logits = logits_flat.view(T1, B1, action_dim)
    values = values_flat.view(T1, B1)
    
    # 取前 T 步
    logits_t = logits[:-1]
    values_t = values
    log_probs_t = F.log_softmax(logits_t, dim=-1)
    
    actions_expanded = actions.unsqueeze(-1)
    logp_t = torch.gather(log_probs_t, 2, actions_expanded).squeeze(-1)
    
    # V-Trace
    vs, pg_adv = vtrace(logp_b, logp_t, rewards, values_t, dones, gamma=GAMMA)
    
    # Loss
    policy_loss = -(pg_adv * logp_t).mean()
    v_pred = values_t[:-1]
    value_loss = F.mse_loss(v_pred, vs)
    entropy = -(torch.exp(log_probs_t) * log_probs_t).sum(-1).mean()
    
    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()
    
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

# ---------------------------
#  Evaluate
# ---------------------------
@torch.no_grad()
def evaluate(env_id, agent, episodes=5):
    env = make_atari_env(env_id)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total_r = False, 0.0
        # 评估时也要带脑子
        core_state = agent.get_initial_state(1, DEVICE)
        while not done:
            a, _, core_state = select_action(agent, obs, core_state, DEVICE)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total_r += r
        scores.append(total_r)
    return float(np.mean(scores))

# ---------------------------
#  Main
# ---------------------------
def run(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = make_atari_env(ENV_ID, seed=seed)
    agent = ImpalaCNN_LSTM(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]
    
    procs = []
    init_w = {k: v.cpu() for k, v in agent.state_dict().items()}
    print("Starting Actors with LSTM...")
    for r in range(NUM_ACTORS):
        param_queues[r].put(init_w)
        p = mp.Process(target=actor_process, args=(r, ENV_ID, UNROLL_LENGTH, data_queue, param_queues[r], seed+r))
        p.daemon=True; p.start(); procs.append(p)
    
    total_frames = 0; upd = 0; start = time.time(); records = []
    os.makedirs("checkpoints", exist_ok=True)
    npy_path = f"checkpoints/impala_lstm_{ENV_ID.replace('/', '_')}_records.npy"
    model_path = f"checkpoints/impala_lstm_{ENV_ID.replace('/', '_')}.pth"

    try:
        while total_frames < MAX_FRAMES:
            batch_list = []
            for _ in range(BATCH_UNROLLS):
                batch = data_queue.get()
                batch_list.append(batch)
                total_frames += UNROLL_LENGTH * FRAME_SKIP
                if "real_scores" in batch and len(batch["real_scores"]) > 0:
                    for score in batch["real_scores"]:
                        records.append((total_frames, score))
                        print(f"Episode finished | Frames: {total_frames} | Score: {score:.2f}")
                        
            loss, pl, vl, ent = learner_train_step(agent, optimizer, batch_list)
            upd += 1
            
            if upd % 10 == 0:
                cpu_w = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try: q.get_nowait()
                    except queue.Empty: pass
                    q.put(cpu_w)
                    
            if upd % 100 == 0:
                fps = total_frames / (time.time() - start)
                np.save(npy_path, np.array(records))
                print(f"[Upd {upd:5d}] F: {total_frames/1e6:.2f}M | Loss: {loss:.3f} (H {ent:.3f}) | FPS: {fps:.0f}")
            
            if upd % 500 == 0:
                sc = evaluate(ENV_ID, agent, episodes=3)
                print(f"=== Eval: {sc:.2f} ===")

    except KeyboardInterrupt: print("Stop.")
    finally:
        for p in procs: p.terminate()
        torch.save(agent.state_dict(), model_path)
        np.save(npy_path, np.array(records))

if __name__ == "__main__":
    run()