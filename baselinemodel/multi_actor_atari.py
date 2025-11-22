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

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- 设备检测 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 开启 CuDNN 加速，针对 CNN 固定输入尺寸优化明显
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True 

# --- 超参数 ---
ITERATION_NUMS = 50000
SAMPLE_NUMS = 64        # 适当增加采样长度，减少通信频率
LR = 0.001
GAMMA = 0.99
CLIP_GRAD_NORM = 0.5
NUM_ACTORS = 4          # 你的 CPU 核心数够多吗？如果卡顿严重，尝试改为 2
ENV_ID = "ALE/Tennis-v5"

# ================================
#   CNN 模型 (结构不变)
# ================================
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # x 进来时已经是 float 且归一化过的
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# ================================
#   [关键修改] 环境构造
# ================================
def make_atari_env(env_id):
    # 显式指定 frameskip=1 以配合 AtariPreprocessing
    env = gym.make(env_id, frameskip=1)
    
    env = gym.wrappers.AtariPreprocessing(
        env, 
        screen_size=84, 
        grayscale_obs=True, 
        scale_obs=False,  # <--- [重点] 这里设为 False！保持 0-255 的 uint8
        frame_skip=4
    )
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
#   [关键修改] 动作选择
# ================================
@torch.no_grad()
def choose_action(agent, state, device):
    # state 是 (4, 84, 84) 的 uint8 numpy array
    if not isinstance(state, torch.Tensor):
        state = np.array(state)
        # 转 tensor -> float -> 归一化 -> 增加 batch 维度
        state_t = torch.tensor(state, dtype=torch.float32).to(device) / 255.0
        state_t = state_t.unsqueeze(0)
    else:
        state_t = state.float().to(device) / 255.0
        if state_t.ndim == 3:
            state_t = state_t.unsqueeze(0)

    logits, v = agent(state_t)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    act = m.sample()
    logp = m.log_prob(act)
    return act.item(), v.item(), logp.item()

# ================================
#   Worker Process
# ================================
def worker_process(id, env_id, state_channels, action_dim, data_queue, weight_queue):
    # [重要] 限制单核，防止 CPU 竞争
    torch.set_num_threads(1)
    
    device = torch.device("cpu")
    task = make_atari_env(env_id)
    agent = ActorCritic(state_channels, action_dim).to(device)
    
    state, _ = task.reset(seed=id + 1000) 
    
    while True:
        try:
            if not weight_queue.empty():
                weights = weight_queue.get_nowait()
                agent.load_state_dict(weights)
        except queue.Empty:
            pass

        states, actions, advs, v_targets = [], [], [], []
        rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

        for _ in range(SAMPLE_NUMS):
            # 存入 Buffer 的是 uint8 (节省内存)
            state_np = np.array(state, dtype=np.uint8)
            states.append(state_np)
            
            # 推理时内部会 / 255.0
            act, v_t, logp_behavior = choose_action(agent, state_np, device)
            actions.append(act)

            next_state, reward, terminated, truncated, _ = task.step(act)
            done = terminated or truncated

            # 计算 Target
            next_state_np = np.array(next_state, dtype=np.uint8)
            # 转换维度和归一化
            next_state_t = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            
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

        if len(actions) > 0:
            batch_data = {
                'states': np.array(states, dtype=np.uint8), # 传输 uint8 !!! 速度关键
                'actions': np.array(actions),
                'advs': np.concatenate(advs) if len(advs) > 0 else np.array([]),
                'v_targets': np.concatenate(v_targets) if len(v_targets) > 0 else np.array([])
            }
            data_queue.put(batch_data)

# ================================
#   Learner 训练函数
# ================================
def train(agent, optim, scheduler, batch):
    # 取出数据，放到 GPU，转为 float，除以 255.0
    states = torch.tensor(batch['states']).to(DEVICE).float() / 255.0
    actions = torch.tensor(batch['actions'], dtype=torch.int64).to(DEVICE)
    advs = torch.tensor(batch['advs'], dtype=torch.float32).to(DEVICE)
    v_targets = torch.tensor(batch['v_targets'], dtype=torch.float32).to(DEVICE)

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
    # 测试环境也要保持一致，scale_obs=False
    env = gym.make(env_id, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, scale_obs=False, frame_skip=4
    )
    task = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    state, _ = task.reset()
    total_reward = 0
    
    for _ in range(3000):
        # 同样需要 uint8 -> float -> / 255.0
        act, _, _ = choose_action(agent, np.array(state, dtype=np.uint8), DEVICE)
        state, reward, terminated, truncated, _ = task.step(act)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

# ================================
#   主流程
# ================================
# ... (前面的 Import, ActorCritic, Worker 等保持不变) ...

# ================================
#   [修改] 主流程 run 函数
# ================================
def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    temp_env = make_atari_env(ENV_ID)
    INPUT_CHANNELS = temp_env.observation_space.shape[0] 
    ACTION_DIM = temp_env.action_space.n
    temp_env.close()

    print(f"Env: {ENV_ID}, Channels: {INPUT_CHANNELS}, Actions: {ACTION_DIM}")

    global_agent = ActorCritic(INPUT_CHANNELS, ACTION_DIM).to(DEVICE)
    
    optim = torch.optim.Adam(global_agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=ITERATION_NUMS)

    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2) 
    weight_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    print(f"Starting {NUM_ACTORS} CPU workers...")
    
    for i in range(NUM_ACTORS):
        p = mp.Process(
            target=worker_process, 
            args=(i, ENV_ID, INPUT_CHANNELS, ACTION_DIM, data_queue, weight_queues[i])
        )
        p.daemon = True
        p.start()
        processes.append(p)

    # --- [修改] 初始化记录列表和帧数计数器 ---
    records = []  # 结构: [[frame_count, score], [frame_count, score], ...]
    total_env_frames = 0 # 记录环境总帧数
    FRAME_SKIP = 4       # 你的环境设置是 frame_skip=4
    
    print("Training started on GPU...")
    start_time = time.time()

    try:
        for i_iter in range(ITERATION_NUMS):
            batch_data = data_queue.get()
            
            # --- [修改] 更新总帧数 ---
            # 每次取出的 batch 包含 SAMPLE_NUMS 个动作
            # 每个动作对应 frame_skip 帧
            batch_size = len(batch_data['actions'])
            total_env_frames += batch_size * FRAME_SKIP
            
            loss = train(global_agent, optim, scheduler, batch_data)

            # 同步权重 (CPU)
            cpu_state_dict = {k: v.cpu() for k, v in global_agent.state_dict().items()}
            for wq in weight_queues:
                try:
                    while not wq.empty():
                        wq.get_nowait()
                except:
                    pass
                wq.put(cpu_state_dict)

            if i_iter % 10 == 0: 
                elapsed = time.time() - start_time
                # 测试
                score = test(ENV_ID, global_agent)
                
                # --- [修改] 记录帧数和分数 ---
                records.append([total_env_frames, score])
                
                print(f"Iter: {i_iter}, Frames: {total_env_frames/1e6:.2f}M, Loss: {loss:.4f}, Score: {score:.2f}, Time: {elapsed:.1f}s")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in processes:
            p.terminate()
            p.join()

    return records

if __name__ == "__main__":
    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    all_results = []
    res = run(seed=42)
    all_results.append(res)
    np.save(f"atari_tennis_fast_{date}.npy", all_results)
    print("Done.")