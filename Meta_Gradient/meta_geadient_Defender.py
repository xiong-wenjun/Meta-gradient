import os
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

# Register ALE environments
gym.register_envs(ale_py)

# Multiprocessing setup
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------------------
#  Global Hyperparameters (A100 Optimized + Meta-Gradient)
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_ID = "ALE/Defender-v5"

# System Params
NUM_ACTORS = 80             # High parallelism for CPU
UNROLL_LENGTH = 20          # Length of each trajectory
BATCH_UNROLLS = 32          # Standard batch size (Meta-Grad needs 2x batches, so effectively 64)
MAX_FRAMES = 50_000_000     # Run longer for meta-learning to shine

# Algorithm Params
LR = 5e-4                   # Inner Loop LR (Policy/Value)
META_LR = 1e-4              # Outer Loop LR (Gamma/Lambda) - usually smaller
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
        obs, _, terminated, truncated, _ = self.env.step(1) # Fire
        if terminated or truncated: self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(0) # Noop
        if terminated or truncated: self.env.reset(**kwargs)
        return obs, info

def make_atari_env(env_id, seed=None):
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, scale_obs=False, 
        frame_skip=FRAME_SKIP, noop_max=30
    )
    env = FireResetWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    return env

# ---------------------------
#  Meta-Gradient Network Structure
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
        self.block1 = ImpalaBlock(input_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        
        self.fc = nn.Linear(32 * 11 * 11, 256)
        self.relu = nn.ReLU(inplace=True)
        
        # Standard Heads
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        
        # === Meta-Gradient Heads (Learnable Hyperparameters) ===
        # These output a value per state, passed through Sigmoid to be in [0, 1]
        self.gamma_head = nn.Linear(256, 1)
        self.lambda_head = nn.Linear(256, 1)

    def forward(self, x):
        # Normal Forward Pass
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        feat = self.relu(self.fc(x))
        
        logits = self.policy(feat)
        value = self.value(feat).squeeze(-1)
        
        # Meta Params
        gamma = torch.sigmoid(self.gamma_head(feat)).squeeze(-1)
        lamb = torch.sigmoid(self.lambda_head(feat)).squeeze(-1)
        
        return logits, value, gamma, lamb

# ---------------------------
#  Functional Forward for Meta-Update (Stateless)
# ---------------------------
# We need this to run the network using temporary "fast weights"
def functional_forward(model, params, x):
    # Extract weights from dict and manually run the architecture
    # This is a simplified version assuming the specific structure above
    
    def conv2d(inp, w, b, stride=1, padding=0):
        return F.conv2d(inp, w, b, stride=stride, padding=padding)
    
    def linear(inp, w, b):
        return F.linear(inp, w, b)
    
    def block_forward(inp, prefix, in_c, out_c):
        # Conv
        x = F.relu(conv2d(inp, params[f'{prefix}.conv.weight'], params[f'{prefix}.conv.bias'], padding=1))
        # Pool
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        # Res1
        r = F.relu(conv2d(x, params[f'{prefix}.res1.conv1.weight'], params[f'{prefix}.res1.conv1.bias'], padding=1))
        r = conv2d(r, params[f'{prefix}.res1.conv2.weight'], params[f'{prefix}.res1.conv2.bias'], padding=1)
        x = F.relu(x + r)
        # Res2
        r = F.relu(conv2d(x, params[f'{prefix}.res2.conv1.weight'], params[f'{prefix}.res2.conv1.bias'], padding=1))
        r = conv2d(r, params[f'{prefix}.res2.conv2.weight'], params[f'{prefix}.res2.conv2.bias'], padding=1)
        x = F.relu(x + r)
        return x

    x = block_forward(x, 'block1', 4, 16)
    x = block_forward(x, 'block2', 16, 32)
    x = block_forward(x, 'block3', 32, 32)
    
    x = x.reshape(x.size(0), -1)
    feat = F.relu(linear(x, params['fc.weight'], params['fc.bias']))
    
    logits = linear(feat, params['policy.weight'], params['policy.bias'])
    value = linear(feat, params['value.weight'], params['value.bias']).squeeze(-1)
    
    # Meta params usually don't need functional forward for validation loss
    # but we include them for completeness
    gamma = torch.sigmoid(linear(feat, params['gamma_head.weight'], params['gamma_head.bias'])).squeeze(-1)
    lamb = torch.sigmoid(linear(feat, params['lambda_head.weight'], params['lambda_head.bias'])).squeeze(-1)
    
    return logits, value, gamma, lamb

# ---------------------------
#  V-Trace with Dynamic Gamma/Lambda
# ---------------------------
def meta_vtrace(behavior_logp, target_logp, rewards, values, dones, gammas, lambdas, rho_bar=1.0, c_bar=1.0):
    """
    V-trace that accepts dynamic gamma and lambda vectors [T, B]
    """
    log_rhos = target_logp - behavior_logp
    rhos = torch.exp(log_rhos)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Use dynamic gamma
    # gammas: [T+1, B] -> We use gammas[t] for step t
    # discounts = gamma * (1-dones)
    # Note: values, gammas, lambdas usually have T+1 length
    
    T, B = rewards.shape
    vs = torch.zeros_like(values)
    vs[-1] = values[-1]

    # Backward Pass
    for t in reversed(range(T)):
        # Dynamic params at time t
        g_t = gammas[t] * (1.0 - dones[t])
        l_t = lambdas[t]
        
        delta = rhos_clipped[t] * (
            rewards[t] + g_t * values[t + 1] - values[t]
        )
        # V-trace recursive formula with lambda
        # Standard V-trace assumes lambda=1.0 for the correction term
        # Meta-Gradient RL paper suggests applying lambda to the correction as well
        vs[t] = values[t] + delta + g_t * l_t * cs[t] * (vs[t + 1] - values[t + 1])

    vs_t = vs[:-1]
    vs_tp1 = vs[1:]
    
    # Policy Gradient Advantage
    # The paper often uses standard advantage or V-trace advantage
    # Here we use the V-trace definition
    pg_adv = rhos_clipped * (rewards + gammas[:-1] * (1.0 - dones) * vs_tp1 - values[:-1])

    return vs_t, pg_adv

# ---------------------------
#  Actor Process
# ---------------------------
@torch.no_grad()
def select_action(agent, obs, device):
    obs = np.array(obs, copy=False)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    if obs_t.ndim == 3: obs_t = obs_t.unsqueeze(0)
    logits, _, _, _ = agent(obs_t) # Ignore value/meta output for acting
    dist = Categorical(logits=logits)
    action = dist.sample()
    logp = dist.log_prob(action)
    return int(action.item()), float(logp.item())

def actor_process(rank, env_id, unroll_length, data_queue, param_queue, seed):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    env = make_atari_env(env_id, seed=seed)
    obs, _ = env.reset()
    action_dim = env.action_space.n
    agent = MetaImpalaCNN(obs.shape[0], action_dim).to(device)

    # Init sync
    try:
        weights = param_queue.get(timeout=10.0)
        agent.load_state_dict(weights)
    except queue.Empty: return

    while True:
        try:
            while True: weights = param_queue.get_nowait()
        except queue.Empty: pass
        agent.load_state_dict(weights)

        obs_list, actions, rewards, dones, behavior_logps = [], [], [], [], []
        for t in range(unroll_length):
            obs_list.append(np.array(obs, dtype=np.uint8))
            a, logp = select_action(agent, obs, device)
            actions.append(a)
            behavior_logps.append(logp)
            obs_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            r = float(np.clip(r, -1, 1))
            rewards.append(r)
            dones.append(1.0 if done else 0.0)
            obs = obs_next
            if done: obs, _ = env.reset()

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
#  Prepare Batch Helper
# ---------------------------
def prepare_batch(batch_list, device):
    obs = torch.tensor(np.stack([b["obs"] for b in batch_list], axis=1), dtype=torch.float32, device=device) / 255.0
    actions = torch.tensor(np.stack([b["actions"] for b in batch_list], axis=1), dtype=torch.long, device=device)
    rewards = torch.tensor(np.stack([b["rewards"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    dones = torch.tensor(np.stack([b["dones"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    logp_b = torch.tensor(np.stack([b["logp_b"] for b in batch_list], axis=1), dtype=torch.float32, device=device)
    return obs, actions, rewards, dones, logp_b

# ---------------------------
#  Meta-Learner Training Step
# ---------------------------
def meta_train_step(agent, optimizer, batch_train, batch_valid):
    """
    batch_train: List of rollouts for Inner Loop (Theta Update)
    batch_valid: List of rollouts for Outer Loop (Meta Update)
    """
    # === 1. Prepare Data ===
    obs_t, act_t, rew_t, done_t, logp_b_t = prepare_batch(batch_train, DEVICE)
    obs_v, act_v, rew_v, done_v, logp_b_v = prepare_batch(batch_valid, DEVICE)
    
    T, B = act_t.shape
    
    # === 2. Inner Loop (Compute Gradients for Theta) ===
    # Forward pass on Training Batch
    T1, B1, C, H, W = obs_t.shape
    logits_flat, values_flat, gammas_flat, lambdas_flat = agent(obs_t.view(T1*B1, C, H, W))
    
    logits = logits_flat.view(T1, B1, -1)
    values = values_flat.view(T1, B1)
    gammas = gammas_flat.view(T1, B1)
    lambdas = lambdas_flat.view(T1, B1)
    
    # Compute Loss using DYNAMIC Gamma/Lambda
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    logp = torch.gather(log_probs, 2, act_t.unsqueeze(-1)).squeeze(-1)
    
    # Calculate Targets (V-Trace)
    # Important: vs and adv depend on gammas and lambdas
    vs, pg_adv = meta_vtrace(
        logp_b_t, logp, rew_t, values, done_t, 
        gammas, lambdas # Pass the network outputs here!
    )
    
    # Standard Loss Components
    # Note: We detach targets to stabilize training, but 
    # the paper sometimes allows gradients flow through targets for meta-learning.
    # Here we stick to standard V-trace stability: targets are detached.
    #policy_loss = -(pg_adv.detach() * logp).mean()
    policy_loss = -(pg_adv.detach() * logp).mean()
    value_loss = F.mse_loss(values[:-1], vs)
    entropy = -(torch.exp(log_probs) * log_probs).sum(-1).mean()
    
    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    
    # === 3. Meta-Update (The Magic) ===
    # We want to differentiate through the update step w.r.t gamma/lambda parameters.
    # This requires second-order derivatives (Hessian-vector products).
    
    # Compute gradients of total_loss w.r.t agent parameters
    # create_graph=True is ESSENTIAL for meta-learning
    # 计算所有参数的梯度
    grads = torch.autograd.grad(total_loss, agent.parameters(), create_graph=True, allow_unused=True)
    
    fast_weights = OrderedDict()
    for (name, param), grad in zip(agent.named_parameters(), grads):
        # 关键逻辑：
        # 1. 如果是 gamma_head/lambda_head，我们在 Inner Loop 保持不动 (或者 grad 是 None)
        # 2. 如果是 Policy/Value 层，我们用包含 meta-info 的梯度去更新它
        
        if grad is None or "gamma_head" in name or "lambda_head" in name:
            fast_weights[name] = param  # 保持原样，不更新
        else:
            fast_weights[name] = param - LR * grad  # 更新，这个 fast_weights 现在携带了 gamma 的梯度信息

    # === 4. Outer Loop (Validation) ===
    # Evaluate Theta' on the Validation Batch
    # We use functional_forward to use the temporary fast_weights
    T_v, B_v = act_v.shape
    obs_v_flat = obs_v.view((T_v+1)*B_v, C, H, W)
    
    # Forward pass using NEW weights
    logits_val, _, _, _ = functional_forward(agent, fast_weights, obs_v_flat)
    
    logits_val = logits_val.view(T_v+1, B_v, -1)
    log_probs_val = F.log_softmax(logits_val[:-1], dim=-1)
    logp_val = torch.gather(log_probs_val, 2, act_v.unsqueeze(-1)).squeeze(-1)
    
    # Meta Objective: Maximize expected return on validation set
    # Simplified proxy: Maximize Probability of actions taken in validation set (Behavior Cloning style)
    # Or minimize standard Policy Gradient Loss on validation set.
    # The paper uses V-trace on validation set too, but for simplicity/stability, 
    # we use a standard Policy Gradient loss on validation set using FIXED gamma=0.99 for the target
    # to anchor the meta-learning.
    
    # Anchor validation targets (Fixed Gamma)
    with torch.no_grad():
        _, val_values_fixed, _, _ = functional_forward(agent, fast_weights, obs_v_flat)
        val_values_fixed = val_values_fixed.view(T_v+1, B_v)
        # Simple n-step return for validation signal
        # This provides a ground-truth signal: "Did the parameter update actually help get more reward?"
        rets = torch.zeros_like(rew_v)
        R = val_values_fixed[-1]
        for t in reversed(range(T_v)):
            R = rew_v[t] + 0.99 * (1 - done_v[t]) * R
            rets[t] = R
        adv_val = rets - val_values_fixed[:-1]

    # Meta Loss: Minimize policy loss on validation set
    # Gradients will flow back from here -> fast_weights -> grads -> total_loss -> Gamma/Lambda Heads
    meta_loss = -(adv_val * logp_val).mean()
    
    # Update Model
    optimizer.zero_grad()
    
    # This backward computes gradients for EVERYTHING (Theta and Meta-Params)
    # The Theta gradients are dominated by the inner loop, Meta-Params by outer loop.
    # Ideally we use separate optimizers, but adding them is a standard simplification (first-order approx)
    # For true meta-learning, we should only apply meta_loss gradients to meta-heads.
    
    # Strategy: 
    # 1. Apply Inner Loop Update (Actual Training)
    # 2. Add Meta-Gradients to Meta-Heads
    
    # We already computed 'grads' for the inner loop. 
    # Let's verify we can backprop meta_loss to the original params.
    meta_grads = torch.autograd.grad(meta_loss, agent.parameters(), retain_graph=True, allow_unused=True)
    
    # Apply Updates
    for param, g_inner, g_meta in zip(agent.parameters(), grads, meta_grads):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        
        # Standard Update
        if g_inner is not None:
            param.grad.add_(g_inner.detach()) # Detach to stop graph
        
        # Meta Update (Only affects Gamma/Lambda heads essentially)
        if g_meta is not None:
            # Scale meta-gradient if needed
            param.grad.add_(g_meta) 

    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optimizer.step()
    
    # Cleanup to save memory
    del grads, fast_weights, meta_grads
    
    return total_loss.item(), gammas.mean().item(), lambdas.mean().item()

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
        while not done:
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
            logits, _, _, _ = agent(obs_t)
            a = torch.argmax(logits, dim=-1).item()
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total_r += r
        scores.append(total_r)
    return float(np.mean(scores))

# ---------------------------
#  Main Loop
# ---------------------------
def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = make_atari_env(ENV_ID, seed=seed)
    obs, _ = env.reset()
    action_dim = env.action_space.n
    env.close()

    print(f"[META-IMPALA] Env: {ENV_ID}, Actions: {action_dim}")
    print(f"[META-IMPALA] Device: {DEVICE}, Actors: {NUM_ACTORS}")

    agent = MetaImpalaCNN(obs.shape[0], action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2)
    param_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    init_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    
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

    npy_path = f"checkpoints/impala_{ENV_ID.replace('/', '_')}_records.npy"
    model_path = f"checkpoints/impala_{ENV_ID.replace('/', '_')}.pth"

    try:
        while total_frames < MAX_FRAMES:
            # Collect 2x Batch Size (Half for Train, Half for Validation)
            full_batch_list = []
            # We need 2 * BATCH_UNROLLS
            required_batches = BATCH_UNROLLS * 2 
            
            for _ in range(required_batches):
                full_batch_list.append(data_queue.get())
                total_frames += UNROLL_LENGTH * FRAME_SKIP

            # Split into Train / Valid
            batch_train = full_batch_list[:BATCH_UNROLLS]
            batch_valid = full_batch_list[BATCH_UNROLLS:]

            loss, avg_gamma, avg_lambda = meta_train_step(agent, optimizer, batch_train, batch_valid)
            update_count += 1

            if update_count % 10 == 0:
                state_dict_cpu = {k: v.cpu() for k, v in agent.state_dict().items()}
                for q in param_queues:
                    try:
                        while not q.empty(): q.get_nowait()
                    except queue.Empty: pass
                    q.put(state_dict_cpu)

            if update_count % 100 == 0:
                fps = total_frames / (time.time() - start_time)
                print(
                    f"[Upd {update_count:5d}] Frames: {total_frames/1e6:.2f}M | "
                    f"Loss: {loss:.3f} | G: {avg_gamma:.3f} L: {avg_lambda:.3f} | FPS: {fps:.0f}"
                )

            if update_count % 500 == 0: # Low freq eval for speed
                avg_score = evaluate(ENV_ID, agent, episodes=3)
                records.append((total_frames, avg_score))
                print(f"=== Eval @ {total_frames/1e6:.2f}M | Score: {avg_score:.2f} ===")
                np.save(npy_path, np.array(records))
                print(f"Running data saved to {npy_path}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for p in processes: p.terminate()
    
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 保存最终的曲线数据
    np.save(npy_path, np.array(records))
    print(f"Final training curve saved to {npy_path}")

if __name__ == "__main__":
    run(seed=42)