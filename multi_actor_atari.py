import gymnasium as gym
import numpy as np
import random
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
import queue
import ale_py

gym.register_envs(ale_py)
# --- 防止 Linux 下 CUDA 初始化报错 ---
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- 超参数调整 ---
ITERATION_NUMS = 10000
SAMPLE_NUMS = 128
LR = 0.0002
GAMMA = 0.99
CLIP_GRAD_NORM = 0.5
NUM_ACTORS = 4
ENV_ID = "ALE/Tennis-v5"

# ================================
#   CNN 模型 (保持不变)
# ================================
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCritic, self).__init__()
        # 卷积层处理图像 (Batch, 4, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 计算卷积后的尺寸: 84->20->9->7. 64通道 * 7 * 7 = 3136
        self.fc = nn.Linear(3136, 512)
        
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# ================================
#   [修改点] 环境构造函数
#   使用 FrameStackObservation
# ================================
def make_atari_env(env_id):
    env = gym.make(env_id, frameskip=1)
    
    # 1. 标准预处理: Resize 84x84, Grayscale, Scale 0-1
    env = gym.wrappers.AtariPreprocessing(
        env, 
        screen_size=84, 
        grayscale_obs=True, 
        scale_obs=True
    )
    
    # 2. [修改] 使用 FrameStackObservation 替代 FrameStack
    # 这会将 state 变为 (4, 84, 84)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    return env

# ================================
#   V-trace 计算
# ================================
def vtrace_calculator(rewards, values, next_values, logp_b, logp_t, dones, gamma):
    T = len(rewards)
    rho = torch.exp(logp_t - logp_b)
    rho_clipped = torch.clamp(rho, max=1.0)
    c = torch.clamp(rho, max=1.0)

    v_trace = torch.zeros(T)
    adv = torch.zeros(T)
    vs_plus_1 = next_values[-1]

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * dones[t] - values[t]
        delta = rho_clipped[t] * delta
        vs = values[t] + delta + gamma * c[t] * (vs_plus_1 - next_values[t]) * dones[t]
        adv[t] = delta
        v_trace[t] = vs
        vs_plus_1 = vs

    return adv.numpy(), v_trace.numpy()

# ================================
#   Actor 动作选择
# ================================
@torch.no_grad()
def choose_action(agent, state):
    # state 可能是 numpy array (4, 84, 84)
    # 转换为 tensor 并增加 batch 维度 -> (1, 4, 84, 84)
    if not isinstance(state, torch.Tensor):
        state = np.array(state) 
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    else:
        state_t = state.unsqueeze(0)

    logits, v = agent(state_t)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    act = m.sample()
    logp = m.log_prob(act)
    return act.item(), v.item(), logp.item()

# ================================
#   持久化 Worker
# ================================
def worker_process(id, env_id, state_channels, action_dim, data_queue, weight_queue):
    task = make_atari_env(env_id)
    agent = ActorCritic(state_channels, action_dim)
    
    state, _ = task.reset(seed=id + 1000) 
    
    while True:
        # --- 同步权重 ---
        try:
            if not weight_queue.empty():
                weights = weight_queue.get_nowait()
                agent.load_state_dict(weights)
        except queue.Empty:
            pass

        # --- 采集数据 ---
        states, actions, advs, v_targets = [], [], [], []
        rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

        for _ in range(SAMPLE_NUMS):
            state_np = np.array(state)
            states.append(state_np)
            
            act, v_t, logp_behavior = choose_action(agent, state_np)
            actions.append(act)

            next_state, reward, terminated, truncated, _ = task.step(act)
            done = terminated or truncated

            # Target Logits 计算
            next_state_np = np.array(next_state)
            next_state_t = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
            
            logits_target, v_t1 = agent(next_state_t)
            probs_target = F.softmax(logits_target, dim=-1)
            logp_target = torch.log(probs_target[0][act] + 1e-8)

            reward = np.clip(reward, -1, 1)

            rewards.append(reward)
            values.append(v_t)
            next_values.append(v_t1.item())
            logp_b.append(logp_behavior)
            logp_t.append(logp_target.item())
            dones.append(0.0 if done else 1.0)

            state = next_state

            if done:
                state, _ = task.reset()
                if len(rewards) > 0:
                    adv, vt = vtrace_calculator(
                        torch.tensor(rewards), torch.tensor(values), torch.tensor(next_values),
                        torch.tensor(logp_b), torch.tensor(logp_t), torch.tensor(dones), GAMMA
                    )
                    advs.append(adv)
                    v_targets.append(vt)
                    rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

        if len(rewards) > 0:
            adv, vt = vtrace_calculator(
                torch.tensor(rewards), torch.tensor(values), torch.tensor(next_values),
                torch.tensor(logp_b), torch.tensor(logp_t), torch.tensor(dones), GAMMA
            )
            advs.append(adv)
            v_targets.append(vt)

        # --- 整理数据 ---
        if len(actions) > 0:
            batch_data = {
                'states': np.array(states),
                'actions': np.array(actions),
                'advs': np.concatenate(advs) if len(advs) > 0 else np.array([]),
                'v_targets': np.concatenate(v_targets) if len(v_targets) > 0 else np.array([])
            }
            data_queue.put(batch_data)

# ================================
#   Learner 训练函数
# ================================
def train(agent, optim, scheduler, batch):
    states = torch.tensor(batch['states'], dtype=torch.float32)
    actions = torch.tensor(batch['actions'], dtype=torch.int64)
    advs = torch.tensor(batch['advs'], dtype=torch.float32)
    v_targets = torch.tensor(batch['v_targets'], dtype=torch.float32)

    if len(states) == 0:
        return 0.0

    logits, v = agent(states)
    v = v.view(-1)
    
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions.view(-1, 1)).view(-1)

    loss_policy = -(advs * log_probs_act).mean()
    loss_value = F.mse_loss(v, v_targets)
    loss_entropy = -(probs * log_probs).sum(dim=1).mean()

    loss = loss_policy + 0.5 * loss_value - 0.01 * loss_entropy

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optim.step()
    scheduler.step()
    
    return loss.item()

# ================================
#   测试函数
# ================================
def test(env_id, agent):
    task = make_atari_env(env_id)
    state, _ = task.reset()
    total_reward = 0
    
    for _ in range(3000):
        act, _, _ = choose_action(agent, np.array(state))
        state, reward, terminated, truncated, _ = task.step(act)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

# ================================
#   主流程
# ================================
def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    temp_env = make_atari_env(ENV_ID)
    # FrameStackObservation (4, 84, 84) -> shape[0] 是 4
    INPUT_CHANNELS = temp_env.observation_space.shape[0] 
    ACTION_DIM = temp_env.action_space.n
    temp_env.close()

    print(f"Env: {ENV_ID}, State Channel: {INPUT_CHANNELS}, Action Dim: {ACTION_DIM}")

    global_agent = ActorCritic(INPUT_CHANNELS, ACTION_DIM)
    global_agent.share_memory()
    
    optim = torch.optim.Adam(global_agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=ITERATION_NUMS)

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2) 
    weight_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    print(f"Starting {NUM_ACTORS} persistent workers with FrameStackObservation...")
    
    for i in range(NUM_ACTORS):
        p = mp.Process(
            target=worker_process, 
            args=(i, ENV_ID, INPUT_CHANNELS, ACTION_DIM, data_queue, weight_queues[i])
        )
        p.daemon = True
        p.start()
        processes.append(p)

    results = []
    print("Training started...")

    try:
        for i_iter in range(ITERATION_NUMS):
            batch_data = data_queue.get()
            
            loss = train(global_agent, optim, scheduler, batch_data)

            current_weights = global_agent.state_dict()
            for wq in weight_queues:
                try:
                    while not wq.empty():
                        wq.get_nowait()
                except:
                    pass
                wq.put(current_weights)

            if i_iter % 50 == 0:
                score = test(ENV_ID, global_agent)
                results.append(score)
                print(f"Iter: {i_iter}, Score: {score:.2f}, Loss: {loss:.4f}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in processes:
            p.terminate()
            p.join()

    return results

if __name__ == "__main__":
    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    all_results = []
    res = run(seed=42)
    all_results.append(res)
    np.save(f"atari_tennis_vtrace_obs_stack_{date}.npy", all_results)
    print("Done.")