import gymnasium as gym
import numpy as np
import random
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
from models import ActorCritic
import queue  # 用于捕获 queue.Empty 异常

# --- 防止 Linux 下 CUDA 初始化报错 ---
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# --- 超参数 ---
ITERATION_NUMS = 5000
SAMPLE_NUMS = 100
LR = 0.001
GAMMA = 0.99
CLIP_GRAD_NORM = 1.0
NUM_ACTORS = 4

# ================================
#   V-trace 计算 (保持不变)
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
    state_t = torch.tensor(state, dtype=torch.float32)
    logits, v = agent(state_t)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    act = m.sample()
    logp = m.log_prob(act)
    return act.item(), v.item(), logp.item()

# ================================
#   持久化 Worker (核心修改)
# ================================
def worker_process(id, gym_name, state_dim, action_dim, data_queue, weight_queue):
    # 1. 每个 Worker 拥有独立的本地环境和模型
    task = gym.make(gym_name)
    agent = ActorCritic(state_dim, action_dim)
    
    # 2. 只 Reset 一次！
    state, _ = task.reset(seed=id) 
    
    # 3. 死循环，一直工作
    while True:
        # --- 同步权重 ---
        # 尝试从队列获取最新权重，如果队列空则继续用旧的（非阻塞或微小阻塞）
        try:
            # 这里的逻辑是：每次采集前检查一下有没有新权重
            # 使用 get_nowait 避免死锁，或者设置很短的 timeout
            if not weight_queue.empty():
                weights = weight_queue.get_nowait()
                agent.load_state_dict(weights)
        except queue.Empty:
            pass

        # --- 采集数据 ---
        states, actions, advs, v_targets = [], [], [], []
        rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

        for _ in range(SAMPLE_NUMS):
            states.append(state)
            
            # 动作选择
            act, v_t, logp_behavior = choose_action(agent, state)
            actions.append(act)

            # 环境交互
            # 这里不需要 try-catch，因为我们保证了只 reset 一次，step 是安全的
            next_state, reward, terminated, truncated, _ = task.step(act)
            done = terminated or truncated

            # 计算 Target Logits 用于 V-trace
            next_state_t = torch.tensor(next_state, dtype=torch.float32)
            logits_target, v_t1 = agent(next_state_t)
            probs_target = F.softmax(logits_target, dim=-1)
            logp_target = torch.log(probs_target[act] + 1e-8)

            rewards.append(reward)
            values.append(v_t)
            next_values.append(v_t1.item())
            logp_b.append(logp_behavior)
            logp_t.append(logp_target.item())
            dones.append(0.0 if done else 1.0)

            state = next_state

            if done:
                state, _ = task.reset()
                # 计算片段 V-trace
                if len(rewards) > 0:
                    adv, vt = vtrace_calculator(
                        torch.tensor(rewards), torch.tensor(values), torch.tensor(next_values),
                        torch.tensor(logp_b), torch.tensor(logp_t), torch.tensor(dones), GAMMA
                    )
                    advs.append(adv)
                    v_targets.append(vt)
                    rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

        # 处理未完成的片段
        if len(rewards) > 0:
            adv, vt = vtrace_calculator(
                torch.tensor(rewards), torch.tensor(values), torch.tensor(next_values),
                torch.tensor(logp_b), torch.tensor(logp_t), torch.tensor(dones), GAMMA
            )
            advs.append(adv)
            v_targets.append(vt)

        # --- 整理数据 ---
        if len(actions) > 0:
            # 转换为 numpy array 避免 concatenate 错误
            batch_data = {
                'states': np.array(states),
                'actions': np.array(actions),
                'advs': np.concatenate(advs) if len(advs) > 0 else np.array([]),
                'v_targets': np.concatenate(v_targets) if len(v_targets) > 0 else np.array([])
            }
            # 发送数据到主进程
            data_queue.put(batch_data)

# ================================
#   Learner 训练函数
# ================================
def train(agent, optim, scheduler, batch):
    states = torch.tensor(batch['states'], dtype=torch.float32)
    actions = torch.tensor(batch['actions'], dtype=torch.int64)
    advs = torch.tensor(batch['advs'], dtype=torch.float32)
    v_targets = torch.tensor(batch['v_targets'], dtype=torch.float32)

    # 检查数据是否对齐，避免空数据报错
    if len(states) == 0 or len(actions) == 0:
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
def test(gym_name, agent):
    task = gym.make(gym_name)
    state, _ = task.reset()
    total_reward = 0
    for _ in range(500):
        act, _, _ = choose_action(agent, state)
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

    gym_name = "CartPole-v1"
    # 获取维度
    temp_env = gym.make(gym_name)
    STATE_DIM = temp_env.observation_space.shape[0]
    ACTION_DIM = temp_env.action_space.n
    temp_env.close()

    # 全局 Agent (Learner)
    global_agent = ActorCritic(STATE_DIM, ACTION_DIM)
    global_agent.share_memory() # 如果使用 Queue 传权重，这行其实可选，但推荐加上
    
    optim = torch.optim.Adam(global_agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=ITERATION_NUMS)

    # 通信队列
    data_queue = mp.Queue(maxsize=NUM_ACTORS * 2) 
    # 为每个 Worker 创建一个权重接收队列（或者用一个广播变量，这里简化用 List of Queues）
    weight_queues = [mp.Queue(maxsize=1) for _ in range(NUM_ACTORS)]

    processes = []
    print(f"Starting {NUM_ACTORS} persistent workers...")
    
    # 1. 启动 Worker 进程 (只启动一次!)
    for i in range(NUM_ACTORS):
        p = mp.Process(
            target=worker_process, 
            args=(i, gym_name, STATE_DIM, ACTION_DIM, data_queue, weight_queues[i])
        )
        p.daemon = True # 设置为守护进程，主进程结束时它们也会结束
        p.start()
        processes.append(p)

    results = []
    print("Training started...")

    # 2. 主循环 (Learner)
    try:
        for i_iter in range(ITERATION_NUMS):
            # 从队列获取数据
            # 这是一个阻塞操作，如果没有数据，Learner 会在这里等待 Actor
            batch_data = data_queue.get()
            
            # 训练
            loss = train(global_agent, optim, scheduler, batch_data)

            # 同步权重给 Worker
            # 我们每隔几次或者是每次训练后广播权重
            current_weights = global_agent.state_dict()
            
            # 广播给所有 Actor
            for wq in weight_queues:
                # 先清空旧权重，防止队列满
                try:
                    while not wq.empty():
                        wq.get_nowait()
                except:
                    pass
                wq.put(current_weights)

            # 打印日志
            if i_iter % 20 == 0:
                score = test(gym_name, global_agent)
                results.append(score)
                print(f"Iter: {i_iter}, Score: {score:.2f}, Loss: {loss:.4f}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # 清理进程
        for p in processes:
            p.terminate()
            p.join()

    return results

if __name__ == "__main__":
    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    all_results = []
    # 跑一个 Seed 验证即可
    res = run(seed=42)
    all_results.append(res)
    np.save(f"baseline_vtrace_multi_actor_{date}.npy", all_results)
    print("Done.")