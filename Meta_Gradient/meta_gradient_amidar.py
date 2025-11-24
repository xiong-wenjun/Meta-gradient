import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 设置

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
import torch.func as tf

# =====================================================
#  Global Config
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

# === [修改 1] 游戏 ID 改为 Amidar ===
ENV_ID = "ALE/Amidar-v5"

NUM_ACTORS = 80
UNROLL_LENGTH = 80
BATCH_UNROLLS = 40
MAX_FRAMES = 200_000_000

LR = 2e-4
META_LR = 1e-4
WARMUP_UPDATES = 50

ENTROPY_COEF = 0.01 
VALUE_COEF = 0.5
CLIP_GRAD_NORM = 40.0
FRAME_SKIP = 4

# =====================================================
#  Meta Parameters (Gamma Only)
# =====================================================

class MetaParams(nn.Module):
    def __init__(self, init_gamma=0.99):
        super().__init__()
        # Gamma 范围 [0.90, 0.999]
        self.min_gamma = 0.95
        self.max_gamma = 0.999
        self.scale = self.max_gamma - self.min_gamma
        
        # 初始化 Logit
        init_val = (init_gamma - self.min_gamma) / self.scale
        init_logit = np.log(init_val / (1.0 - init_val))
        
        self.gamma_logit = nn.Parameter(torch.tensor(float(init_logit)))

    @property
    def gamma(self):
        return self.min_gamma + self.scale * torch.sigmoid(self.gamma_logit)

# =====================================================
#  Wrappers (关键修改)
# =====================================================

class FireResetWrapper(gym.Wrapper):
    """
    [修改 2] 增强版 FireResetWrapper
    Amidar 必须检测到 'FIRE' 并按下才能开始，否则会卡在标题界面。
    """
    def __init__(self, env):
        super().__init__(env)
        # 自动检测动作空间里是否有 'FIRE'
        action_meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in action_meanings:
            self.has_fire = True
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
        
        # 再执行一步 NOOP (动作 0) 让游戏逻辑走一步，防止画面没刷出来
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
#  Model (保持不变)
# =====================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
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
        self.lstm = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.policy = nn.Linear(hidden_size, action_dim)
        self.value = nn.Linear(hidden_size, 1)

    def get_initial_state(self, batch_size, device):
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
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

        h, c = core_state
        
        outputs = []
        for t in range(T):
            h, c = self.lstm(feat[t], (h, c))
            outputs.append(h)
            
        lstm_out = torch.stack(outputs, dim=0)
        
        out_flat = lstm_out.view(T * B, -1)
        logits = self.policy(out_flat)
        values = self.value(out_flat).squeeze(-1)

        return logits, values, (h, c)

# =====================================================
#  V-trace
# =====================================================

def vtrace_meta(behavior_logp, target_logp, rewards, values, dones, gamma_tensor, rho_bar=1.0, c_bar=1.0):
    T, B = rewards.shape

    log_rhos = target_logp - behavior_logp
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    discounts = gamma_tensor * (1.0 - dones)
    values_detached = values.detach()

    vs = torch.zeros_like(values_detached)
    vs[-1] = values_detached[-1]

    for t in reversed(range(T)):
        delta = rhos_clipped[t] * (rewards[t] + discounts[t] * values_detached[t+1] - values_detached[t])
        vs[t] = values_detached[t] + delta + discounts[t] * cs[t] * (vs[t+1] - values_detached[t+1])

    vs_t = vs[:-1]
    vs_tp1 = vs[1:]

    pg_adv = rhos_clipped * (rewards + discounts * vs_tp1 - values_detached[:-1])
    
    return vs_t, pg_adv.detach()

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
        weights = param_queue.get(timeout=60.0)
        agent.load_state_dict(weights)
    except queue.Empty:
        return

    episode_return = 0.0

    while True:
        try:
            while True:
                weights = param_queue.get_nowait()
        except queue.Empty:
            pass
        agent.load_state_dict(weights)

        obs_list, actions, rewards, dones, behavior_logps = [], [], [], [], []
        real_ret = [] 
        
        init_h = core_state[0].cpu().numpy()
        init_c = core_state[1].cpu().numpy()

        for t in range(unroll_length):
            obs_list.append(np.array(obs, dtype=np.uint8))
            a, logp, core_state = select_action(agent, obs, core_state, device)
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            episode_return += r
            
            # 使用 Log-Scaling，Amidar 分数也适合这个
            r_scaled = np.sign(r) * np.log(1.0 + np.abs(r))
            r_clip = float(r_scaled)

            actions.append(a)
            behavior_logps.append(logp)
            rewards.append(r_clip)
            dones.append(1.0 if done else 0.0)

            obs = next_obs
            
            if done:
                real_ret.append(episode_return)
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
            "real_returns": np.array(real_ret, dtype=np.float32),
        }
        
        try:
            data_queue.put(batch)
        except:
            return

# =====================================================
#  Learner
# =====================================================

def compute_loss_stateless(params, buffers, obs, actions, rewards, dones, logp_b, init_h, init_c, 
                           gamma_tensor, agent_ref):
    """
    Ent Coef 使用全局 ENTROPY_COEF
    """
    T, B = actions.shape
    T1 = obs.shape[0]
    
    core_state = (init_h, init_c)
    
    logits_flat, values_flat, _ = tf.functional_call(agent_ref, (params, buffers), (obs, core_state))
    
    action_dim = logits_flat.shape[-1]
    logits = logits_flat.view(T1, B, action_dim)
    values = values_flat.view(T1, B)
    
    logits_t = logits[:-1]
    values_t = values
    log_probs_t = F.log_softmax(logits_t, dim=-1)
    
    actions_expanded = actions.unsqueeze(-1)
    target_logp = torch.gather(log_probs_t, 2, actions_expanded).squeeze(-1)
    
    # Meta-Learning 只针对 Gamma
    vs, pg_adv = vtrace_meta(logp_b, target_logp, rewards, values_t, dones, gamma_tensor)
    
    policy_loss = -(pg_adv * target_logp).mean()
    value_loss = F.mse_loss(values_t[:-1], vs)
    
    probs_t = torch.exp(log_probs_t)
    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()
    
    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    return total_loss, policy_loss, value_loss, entropy

def meta_learner_train_step(agent, meta_params, optimizer, meta_optimizer, batch_list, update_count):
    # ... (前面的数据准备代码 prepare_data 和 batch 分割不变) ...
    mid = 32
    batch_A = batch_list[:mid]
    batch_B = batch_list[mid:]
    
    def prepare_data(b_lst):
        # ... (保持原样) ...
        obs = torch.tensor(np.stack([b["obs"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE) / 255.0
        act = torch.tensor(np.stack([b["actions"] for b in b_lst], axis=1), dtype=torch.long, device=DEVICE)
        rew = torch.tensor(np.stack([b["rewards"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE)
        don = torch.tensor(np.stack([b["dones"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE)
        log = torch.tensor(np.stack([b["logp_b"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE)
        ih = torch.tensor(np.stack([b["init_h"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE).squeeze(0)
        ic = torch.tensor(np.stack([b["init_c"] for b in b_lst], axis=1), dtype=torch.float32, device=DEVICE).squeeze(0)
        return obs, act, rew, don, log, ih, ic

    data_A = prepare_data(batch_A)
    data_B = prepare_data(batch_B)
    
    params = dict(agent.named_parameters())
    buffers = dict(agent.named_buffers())
    
    curr_gamma = meta_params.gamma
    
    # === Phase 1: Inner Loop ===
    loss_A, pl_A, vl_A, ent_A = compute_loss_stateless(
        params, buffers, *data_A, curr_gamma.detach(), agent
    )
    
    grads = torch.autograd.grad(loss_A, params.values(), create_graph=True)
    
    lr_inner = optimizer.param_groups[0]['lr'] # 获取当前动态衰减后的 LR
    params_prime = {}
    for (name, p), g in zip(params.items(), grads):
        params_prime[name] = p - lr_inner * g

# 在 meta_learner_train_step 函数的 Phase 2 部分修改

    # === Phase 2: Outer Loop (Meta Update) ===
    if update_count >= WARMUP_UPDATES:
        loss_B, _, _, _ = compute_loss_stateless(
            params_prime, buffers, *data_B, curr_gamma, agent
        )
        
        # =================================================
        # [新增] Gamma 正则化 (Regularization)
        # =================================================
        # 强制 Gamma 靠近 0.999
        # 系数 0.1 是经验值，如果 Gamma 还在降，可以加大到 0.5
        target_gamma = 0.999
        reg_coef = 0.1  
        
        gamma_reg_loss = reg_coef * (target_gamma - curr_gamma) ** 2
        
        # 总 Loss = 验证集 Loss + 正则化项
        total_meta_loss = loss_B + gamma_reg_loss
        
        meta_optimizer.zero_grad()
        total_meta_loss.backward()
        
        # 稍微放宽一点裁剪，让正则化的梯度能传过去
        torch.nn.utils.clip_grad_norm_(meta_params.parameters(), 1.0)
        meta_optimizer.step()
    
    # === Phase 3: Actual Update ===
    optimizer.zero_grad()
    loss_A_final, _, _, _ = compute_loss_stateless(
        params, buffers, *data_A, curr_gamma.detach(), agent
    )
    loss_A_final.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()
    
    return loss_A.item(), pl_A.item(), vl_A.item(), ent_A.item(), curr_gamma.item()


# =====================================================
#  Evaluation
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
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total_r += r
        scores.append(total_r)
    env.close()
    return float(np.mean(scores)), float(np.std(scores))

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

    print(f"[Meta-IMPALA v2] Fixes: RMSProp + LR Decay + Gamma Reg")

    agent = ImpalaCNN_LSTM(obs.shape[0], action_dim).to(DEVICE)
    
    # === [关键修复 1] 切换到 RMSProp (DeepMind 标配) ===
    # 初始 LR 可以稍微调低一点，RMSProp 比较猛
    LR_START = 4e-4
    LR_END = 1e-6 # 最后不完全降到0，保留一点点适应性
    
    optimizer = torch.optim.RMSprop(
        agent.parameters(),
        lr=LR_START,
        alpha=0.99, # 动量衰减
        eps=0.01,   # [极其重要] 防止分母为0，增加数值稳定性
        momentum=0.0
    )

    meta_params = MetaParams().to(DEVICE)
    meta_optimizer = torch.optim.Adam(meta_params.parameters(), lr=META_LR)

    # ... (中间的 Queue 初始化代码不变) ...
    # ... (Process 启动代码不变) ...
    
    curr_g = meta_params.gamma.item()
    
    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    init_state = {k: v.cpu() for k, v in agent.state_dict().items()}
    
    for rank in range(NUM_ACTORS):
        param_queues[rank].put(init_state)
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
    npy_path = f"checkpoints/meta_impala_v2_gamma_only_amidar_records.npy"
    model_path = f"checkpoints/meta_impala_v2_gamma_only_amidar.pth"
    best_score = -float('inf')

    try:
        while total_frames < MAX_FRAMES:
            # === [关键修复 2] 学习率线性衰减 ===
            progress = total_frames / MAX_FRAMES
            current_lr = LR_START * (1.0 - progress) + LR_END * progress
            
            # 手动更新优化器的 LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            batch_list = []
            for _ in range(BATCH_UNROLLS):
                batch = data_queue.get()
                batch_list.append(batch)
                total_frames += UNROLL_LENGTH * FRAME_SKIP

                if "real_returns" in batch and len(batch["real_returns"]) > 0:
                    for ret in batch["real_returns"]:
                        records.append((total_frames, ret, curr_g))
                        print(f"Episode | Frames: {total_frames} | Return: {ret:.1f} | G: {curr_g:.4f}")

            loss, pl, vl, ent_val, curr_g = meta_learner_train_step(
                agent, meta_params, optimizer, meta_optimizer, batch_list, update_count
            )
            update_count += 1

            if update_count % 5 == 0:
                cpu_state = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try:
                        while not q.empty(): q.get_nowait()
                    except queue.Empty: pass
                    q.put(cpu_state)

            if update_count % 100 == 0:
                fps = total_frames / (time.time() - start_time)
                # 打印一下当前的 LR
                print(
                    f"[Upd {update_count:5d}] F: {total_frames/1e6:.2f}M | "
                    f"Loss: {loss:.2e} | VL: {vl:.2e} | "
                    f"Gamma: {curr_g:.4f} | LR: {current_lr:.2e} | FPS: {fps:.0f}"
                )
                np.save(npy_path, np.array(records, dtype=np.float32))
            
            if update_count % 500 == 0:
                mean_sc, std_sc = evaluate(ENV_ID, agent, episodes=5)
                print(f"=== Eval @ {total_frames/1e6:.2f}M | Score: {mean_sc:.1f} ± {std_sc:.1f} ===")
                
                if mean_sc > best_score:
                    best_score = mean_sc
                    torch.save(agent.state_dict(), model_path.replace('.pth', '_best.pth'))
                    print(f">>> New Best: {best_score:.1f}")

    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        for p in processes:
            p.terminate()
        torch.save(agent.state_dict(), model_path)
        np.save(npy_path, np.array(records, dtype=np.float32))
        print(f"Saved to {model_path}")

if __name__ == "__main__":
    run()