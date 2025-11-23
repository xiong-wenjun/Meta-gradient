import os
# 指定使用第二张显卡 (索引为 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import time
import random
import queue
import numpy as np
from collections import OrderedDict

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
#  Global Hyperparameters (DeepMind Paper Config)
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # A100/3090/4090 开启 TF32 加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ENV_ID = "ALE/Defender-v5"

# --- Paper Implementation Details ---
# Actors & Rollout
NUM_ACTORS = 80             # Paper: 80 (A100 可以轻松带起来)
UNROLL_LENGTH = 20          # Paper: 20
FRAME_SKIP = 4

# Inner Loop (Agent) Params
INNER_BATCH_SIZE = 32       # Paper: 32
LR_START = 0.0006           # Paper: 0.0006
LR_END = 0.0                # Paper: Anneal to 0
RMS_ALPHA = 0.99            # Paper: Decay 0.99
RMS_MOMENTUM = 0.0          # Paper: Momentum 0.0
RMS_EPS = 0.1               # Paper: Epsilon 0.1 (这是防梯度的关键！)
CLIP_GRAD_NORM = 40.0       # Paper: 40.0
ENTROPY_COEF = 0.01         # Paper: 0.01
VALUE_COEF = 0.5            # Paper: 0.5

# Meta Loop (Hyperparams) Params
META_BATCH_SIZE = 8         # Paper: 8
META_LR = 0.001             # Paper: 0.001 (Adam)
EMBED_SIZE = 16             # Paper: Embedding size for eta = 16

MAX_FRAMES = 50_000_000     # Total training frames

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
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc: self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(0)
        if term or trunc: self.env.reset(**kwargs)
        return obs, info

def make_atari_env(env_id, seed=None):
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, scale_obs=False, 
        frame_skip=FRAME_SKIP, noop_max=30
    )
    env = FireResetWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    if seed is not None: env.reset(seed=seed)
    return env

# ---------------------------
#  Network with Embedding Head (Paper Config)
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

class MetaImpalaCNN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super().__init__()
        # Deep ResNet
        self.block1 = ImpalaBlock(input_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        
        self.fc = nn.Linear(32 * 11 * 11, 256)
        self.relu = nn.ReLU(inplace=True)
        
        # Policy & Value
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        
        # === Meta Heads with Embedding (Paper Config) ===
        self.gamma_embed = nn.Linear(256, EMBED_SIZE)
        self.gamma_out = nn.Linear(EMBED_SIZE, 1)
        
        self.lambda_embed = nn.Linear(256, EMBED_SIZE)
        self.lambda_out = nn.Linear(EMBED_SIZE, 1)
        # 1. 初始化 Bias 为 +5.0
        #    这样 Sigmoid(5.0) ≈ 0.993
        #    让 Gamma 和 Lambda 从一开始就处于“高位”
        nn.init.constant_(self.gamma_out.bias, 5.0)
        nn.init.constant_(self.lambda_out.bias, 5.0)
        
        # 2. (可选) 初始化 Weight 为极小值
        #    这样一开始 feature 对 gamma 的影响很小，主要由 bias 决定
        nn.init.orthogonal_(self.gamma_out.weight, gain=0.01)
        nn.init.orthogonal_(self.lambda_out.weight, gain=0.01)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        feat = self.relu(self.fc(x))
        
        logits = self.policy(feat)
        value = self.value(feat).squeeze(-1)
        
        # Meta Gamma (Constrained Range)
        g_emb = self.relu(self.gamma_embed(feat))
        raw_gamma = torch.sigmoid(self.gamma_out(g_emb)).squeeze(-1)
        gamma = raw_gamma * 0.05+ 0.95  # [0.95, 0.999]

        # Meta Lambda (Constrained Range)
        l_emb = self.relu(self.lambda_embed(feat))
        raw_lambda = torch.sigmoid(self.lambda_out(l_emb)).squeeze(-1)
        lamb = raw_lambda * 0.1 + 0.9     # [0.9, 1.0]
        
        return logits, value, gamma, lamb

# ---------------------------
#  Functional Forward (For Meta-Update Validation)
# ---------------------------
def functional_forward(model, params, x):
    def conv2d(inp, w, b, stride=1, padding=0): return F.conv2d(inp, w, b, stride=stride, padding=padding)
    def linear(inp, w, b): return F.linear(inp, w, b)
    def block(inp, pre, in_c, out_c):
        x = F.relu(conv2d(inp, params[f'{pre}.conv.weight'], params[f'{pre}.conv.bias'], padding=1))
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        r = F.relu(conv2d(x, params[f'{pre}.res1.conv1.weight'], params[f'{pre}.res1.conv1.bias'], padding=1))
        r = conv2d(r, params[f'{pre}.res1.conv2.weight'], params[f'{pre}.res1.conv2.bias'], padding=1)
        x = F.relu(x + r)
        r = F.relu(conv2d(x, params[f'{pre}.res2.conv1.weight'], params[f'{pre}.res2.conv1.bias'], padding=1))
        r = conv2d(r, params[f'{pre}.res2.conv2.weight'], params[f'{pre}.res2.conv2.bias'], padding=1)
        x = F.relu(x + r)
        return x

    x = block(x, 'block1', 4, 16)
    x = block(x, 'block2', 16, 32)
    x = block(x, 'block3', 32, 32)
    x = x.reshape(x.size(0), -1)
    feat = F.relu(linear(x, params['fc.weight'], params['fc.bias']))
    logits = linear(feat, params['policy.weight'], params['policy.bias'])
    value = linear(feat, params['value.weight'], params['value.bias']).squeeze(-1)
    return logits, value

# ---------------------------
#  Meta V-Trace
# ---------------------------
def meta_vtrace(logp_b, logp_t, rew, val, done, gammas, lambdas, rho_bar=1.0, c_bar=1.0):
    log_rhos = logp_t - logp_b
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    
    T, B = rew.shape
    vs = torch.zeros_like(val)
    vs[-1] = val[-1]
    
    for t in reversed(range(T)):
        g_t = gammas[t] * (1.0 - done[t])
        l_t = lambdas[t]
        delta = rhos_clipped[t] * (rew[t] + g_t * val[t + 1] - val[t])
        vs[t] = val[t] + delta + g_t * l_t * cs[t] * (vs[t + 1] - val[t + 1])
        
    pg_adv = rhos_clipped * (rew + gammas[:-1] * (1.0 - done) * vs[1:] - val[:-1])
    return vs[:-1], pg_adv

# ---------------------------
#  Helpers
# ---------------------------
@torch.no_grad()
def select_action(agent, obs, device):
    obs = np.array(obs, copy=False)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    if obs_t.ndim == 3: obs_t = obs_t.unsqueeze(0)
    logits, _, _, _ = agent(obs_t)
    dist = Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    return int(action.item()), float(logp.item())

def prepare_batch(batch_list, device):
    obs = torch.tensor(np.stack([b["obs"] for b in batch_list], axis=1), dtype=torch.float32, device=device) / 255.0
    actions = torch.tensor(np.stack([b["actions"] for b in batch_list], axis=1), dtype=torch.long, device=device)
    rewards = torch.tensor(np.stack([b["rewards"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    dones = torch.tensor(np.stack([b["dones"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    logp_b = torch.tensor(np.stack([b["logp_b"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    return obs, actions, rewards, dones, logp_b

def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    agent = MetaImpalaCNN(obs.shape[0], env.action_space.n).to(device)
    
    try: weights = param_queue.get(timeout=10.0)
    except queue.Empty: return
    agent.load_state_dict(weights)

    while True:
        try: 
            while True: weights = param_queue.get_nowait()
        except queue.Empty: pass
        agent.load_state_dict(weights)
        
        obs_l, act_l, rew_l, don_l, log_l = [], [], [], [], []
        for _ in range(unroll_length):
            obs_l.append(np.array(obs, dtype=np.uint8))
            a, logp = select_action(agent, obs, device)
            obs_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            act_l.append(a); log_l.append(logp); rew_l.append(np.clip(r, -1, 1)); don_l.append(1.0 if done else 0.0)
            obs = obs_next
            if done: obs, _ = env.reset()
        obs_l.append(np.array(obs, dtype=np.uint8))
        
        data_queue.put({
            "obs": np.stack(obs_l, axis=0), "actions": np.array(act_l), 
            "rewards": np.array(rew_l), "dones": np.array(don_l), "logp_b": np.array(log_l)
        })

# ---------------------------
#  Meta-Training Step
# ---------------------------
def meta_train_step(agent, optimizer_inner, optimizer_meta, batch_train, batch_valid, current_lr):
    # 1. Prepare Data
    obs_t, act_t, rew_t, done_t, logp_b_t = prepare_batch(batch_train, DEVICE)
    obs_v, act_v, rew_v, done_v, logp_b_v = prepare_batch(batch_valid, DEVICE)
    
    # === Inner Loop ===
    T1, B1, C, H, W = obs_t.shape
    logits_flat, values_flat, gammas_flat, lambdas_flat = agent(obs_t.view(T1*B1, C, H, W))
    
    logits = logits_flat.view(T1, B1, -1)
    values = values_flat.view(T1, B1)
    gammas = gammas_flat.view(T1, B1)
    lambdas = lambdas_flat.view(T1, B1)
    
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    logp = torch.gather(log_probs, 2, act_t.unsqueeze(-1)).squeeze(-1)
    
    # Calculate Targets WITHOUT detach on gammas/lambdas
    vs, pg_adv = meta_vtrace(logp_b_t, logp, rew_t, values, done_t, gammas, lambdas)
    
    # Loss (Allowing gradient flow through Value Loss)
    value_loss = F.mse_loss(values[:-1], vs)
    policy_loss = -(pg_adv.detach() * logp).mean()
    entropy = -(torch.exp(log_probs) * log_probs).sum(-1).mean()
    
    inner_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    # Circuit Breaker 1 (Loss Explosion)
    if inner_loss.item() > 200 or torch.isnan(inner_loss):
        print(f"[WARN] Inner Loss Explosion: {inner_loss.item()}. Skip.")
        optimizer_inner.zero_grad(); optimizer_meta.zero_grad()
        return inner_loss.item(), 0, 0

    # Gradients for Inner Update
    grads = torch.autograd.grad(inner_loss, agent.parameters(), create_graph=True, allow_unused=True)
    
    # Manual SGD Step for Meta-Validation
    fast_weights = OrderedDict()
    for (name, param), grad in zip(agent.named_parameters(), grads):
        if grad is None or "gamma" in name or "lambda" in name:
            fast_weights[name] = param
        else:
            fast_weights[name] = param - current_lr * grad

    # === Outer Loop (Meta Update) ===
    T_v, B_v = act_v.shape
    logits_val, _ = functional_forward(agent, fast_weights, obs_v.view((T_v+1)*B_v, C, H, W))
    
    logits_val = logits_val.view(T_v+1, B_v, -1)
    log_probs_val = F.log_softmax(logits_val[:-1], dim=-1)
    logp_val = torch.gather(log_probs_val, 2, act_v.unsqueeze(-1)).squeeze(-1)
    
    # Fixed Gamma anchor for validation
    with torch.no_grad():
        _, val_values_fixed = functional_forward(agent, fast_weights, obs_v.view((T_v+1)*B_v, C, H, W))
        val_values_fixed = val_values_fixed.view(T_v+1, B_v)
        rets, R = torch.zeros_like(rew_v), val_values_fixed[-1]
        for t in reversed(range(T_v)):
            R = rew_v[t] + 0.99 * (1 - done_v[t]) * R
            rets[t] = R
        adv_val = rets - val_values_fixed[:-1]
        
    meta_loss = -(adv_val * logp_val).mean()

    # Apply Meta Gradients
    optimizer_meta.zero_grad()
    meta_grads = torch.autograd.grad(meta_loss, agent.parameters(), allow_unused=True)
    
    for (name, param), g_meta in zip(agent.named_parameters(), meta_grads):
        if g_meta is not None and ("gamma" in name or "lambda" in name):
            if param.grad is None: param.grad = torch.zeros_like(param)
            param.grad.add_(g_meta)
            
    # Circuit Breaker 2 (Gradient Explosion)
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    if torch.isnan(grad_norm) or grad_norm > 1000:
        print("[WARN] Meta Gradient Explosion. Skip.")
        optimizer_inner.zero_grad(); optimizer_meta.zero_grad()
        return inner_loss.item(), gammas.mean().item(), lambdas.mean().item()
        
    optimizer_meta.step()

    # Apply Inner Gradients (RMSProp)
    optimizer_inner.zero_grad()
    for param, g_inner in zip(agent.parameters(), grads):
        if g_inner is not None:
            if param.grad is None: param.grad = torch.zeros_like(param)
            param.grad.add_(g_inner.detach())
            
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer_inner.step()
    
    return inner_loss.item(), gammas.mean().item(), lambdas.mean().item()

@torch.no_grad()
def evaluate(env_id, agent, episodes=3):
    env = make_atari_env(env_id); scores = []
    for _ in range(episodes):
        obs, _ = env.reset(); done, tr = False, 0.0
        while not done:
            t_obs = torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)/255.0
            logits, _, _, _ = agent(t_obs)
            a = torch.argmax(logits, -1).item()
            obs, r, term, trunc, _ = env.step(a); done = term or trunc; tr += r
        scores.append(tr)
    return np.mean(scores)

# ---------------------------
#  Main
# ---------------------------
def run(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = make_atari_env(ENV_ID, seed=seed)
    agent = MetaImpalaCNN(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    
    # === Optimizer Config from Paper ===
    # Inner: RMSProp (eps=0.1)
    optimizer_inner = torch.optim.RMSprop(
        agent.parameters(), lr=LR_START, alpha=RMS_ALPHA, eps=RMS_EPS, momentum=RMS_MOMENTUM
    )
    # Outer: Adam
    optimizer_meta = torch.optim.Adam(agent.parameters(), lr=META_LR)
    
    # Linear LR Scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer_inner, start_factor=1.0, end_factor=0.001, total_iters=MAX_FRAMES // (INNER_BATCH_SIZE * UNROLL_LENGTH)
    )

    data_queue = mp.Queue(maxsize=NUM_ACTORS*2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]
    init_w = {k: v.cpu() for k,v in agent.state_dict().items()}
    
    print("Starting Actors...")
    procs = []
    for r in range(NUM_ACTORS):
        param_queues[r].put(init_w)
        p = mp.Process(target=actor_process, args=(r, ENV_ID, UNROLL_LENGTH, data_queue, param_queues[r], seed+r))
        p.daemon=True; p.start(); procs.append(p)

    total_frames = 0; upd = 0; start = time.time(); records = []
    
    # ====== 创建保存目录 ======
    os.makedirs("checkpoints", exist_ok=True)
    npy_path = f"checkpoints/meta_defender_recs.npy"
    
    try:
        while total_frames < MAX_FRAMES:
            # Paper: Batch Size 32 (Inner) + 8 (Meta)
            batch_data = []
            for _ in range(INNER_BATCH_SIZE + META_BATCH_SIZE):
                batch_data.append(data_queue.get())
                total_frames += UNROLL_LENGTH * FRAME_SKIP
            
            b_train = batch_data[:INNER_BATCH_SIZE]
            b_valid = batch_data[INNER_BATCH_SIZE:]
            
            curr_lr = scheduler.get_last_lr()[0]
            loss, g, l = meta_train_step(agent, optimizer_inner, optimizer_meta, b_train, b_valid, curr_lr)
            
            scheduler.step()
            upd += 1
            
            if upd % 10 == 0:
                cpu_w = {k: v.cpu() for k,v in agent.state_dict().items()}
                for q in param_queues:
                    try: 
                        while not q.empty(): q.get_nowait()
                    except: pass
                    q.put(cpu_w)
            
            if upd % 100 == 0:
                fps = total_frames / (time.time() - start)
                print(f"[Upd {upd:5d}] F: {total_frames/1e6:.2f}M | L: {loss:.3f} | G: {g:.3f} L: {l:.3f} | LR: {curr_lr:.6f} | FPS: {fps:.0f}")
                
            if upd % 500 == 0:
                sc = evaluate(ENV_ID, agent)
                records.append((total_frames, sc))
                print(f"=== Eval: {sc:.2f} ===")
                # ====== 关键点：每次评估完保存 .npy ======
                np.save(npy_path, np.array(records))
                print(f"Data saved to {npy_path}")

    except KeyboardInterrupt: print("Stop.")
    finally:
        for p in procs: p.terminate()
        torch.save(agent.state_dict(), f"checkpoints/meta_defender_final.pth")
        # ====== 关键点：结束时也保存 .npy ======
        np.save(npy_path, np.array(records))

if __name__ == "__main__":
    run()