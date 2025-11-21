"""
IMPALA-lite baseline
A2C + V-trace
Gymnasium support
"""

import gymnasium as gym
import numpy as np
import random
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic  # 普通 ActorCritic，无需 MetaModule

import matplotlib.pyplot as plt

ITERATION_NUMS = 1000
SAMPLE_NUMS = 50
LR = 0.01
GAMMA = 0.99
CLIP_GRAD_NORM = 40


# ================================
#   V-trace FUNCTION
# ================================
def vtrace_calculator(rewards, values, next_values, logp_b, logp_t, dones, gamma):
    """
    rewards, values, next_values, logp_b, logp_t, dones 均为 Torch 1D tensor
    CartPole 版本，简化版 IMPALA V-trace
    """
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
#      ACTOR SAMPLING
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
#          ROLL-OUT
# ================================
def roll_out(agent, task, sample_nums, init_state, gamma):
    states = []
    actions = []
    advs = []
    v_targets = []

    rewards = []
    values = []
    next_values = []
    logp_b = []
    logp_t = []
    dones = []

    state = init_state

    for i in range(sample_nums):
        states.append(state)

        act, v_t, logp_behavior = choose_action(agent, state)
        actions.append(act)

        next_state, reward, terminated, truncated, info = task.step(act)
        done = terminated or truncated

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

        # 完整 episode 的 end
        if done:
            state, info = task.reset()

            adv, vt = vtrace_calculator(
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(values, dtype=torch.float32),
                torch.tensor(next_values, dtype=torch.float32),
                torch.tensor(logp_b, dtype=torch.float32),
                torch.tensor(logp_t, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32),
                gamma
            )

            advs.append(adv)
            v_targets.append(vt)

            rewards, values, next_values, logp_b, logp_t, dones = [], [], [], [], [], []

    # 处理最后一次未终止的片段
    if len(rewards) > 0:
        adv, vt = vtrace_calculator(
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(next_values, dtype=torch.float32),
            torch.tensor(logp_b, dtype=torch.float32),
            torch.tensor(logp_t, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            gamma
        )
        advs.append(adv)
        v_targets.append(vt)

    advs = np.concatenate(advs)
    v_targets = np.concatenate(v_targets)

    return states, actions, advs, v_targets, state


# ================================
#          TRAIN FUNCTION
# ================================
def train(agent, optim, scheduler, states, actions, advs, v_targets, action_dim):
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)

    advs = torch.tensor(advs, dtype=torch.float32)
    v_targets = torch.tensor(v_targets, dtype=torch.float32)

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)

    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs[range(len(actions)), actions]

    loss_policy = -(advs * log_probs_act).mean()
    loss_value = F.mse_loss(v_targets, v)
    loss_entropy = -(probs * log_probs).sum(dim=1).mean()

    loss = loss_policy + 0.25 * loss_value - 0.001 * loss_entropy

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optim.step()
    scheduler.step()


# ================================
#             TESTING
# ================================
def test(gym_name, agent):
    task = gym.make(gym_name)
    result = 0

    for _ in range(10):
        state, info = task.reset()
        for _ in range(500):
            act, _, _ = choose_action(agent, state)
            next_state, reward, terminated, truncated, info = task.step(act)
            result += reward
            state = next_state
            if terminated or truncated:
                break
    return result


# ================================
#             MAIN LOOP
# ================================
def run(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gym_name = "CartPole-v1"
    task = gym.make(gym_name)
    task.reset(seed=seed)

    STATE_DIM = task.observation_space.shape[0]
    ACTION_DIM = task.action_space.n

    agent = ActorCritic(STATE_DIM, ACTION_DIM)
    optim = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=ITERATION_NUMS)

    init_state, info = task.reset()
    results = []

    for i in range(ITERATION_NUMS):
        states, actions, advs, v_targets, init_state = roll_out(agent, task, SAMPLE_NUMS, init_state, GAMMA)

        train(agent, optim, scheduler, states, actions, advs, v_targets, ACTION_DIM)

        if (i + 1) % (ITERATION_NUMS // 10) == 0:
            score = test(gym_name, agent) / 10.0
            print(f"iteration {i+1}, test score {score}")
            results.append(score)

    return results


if __name__ == "__main__":
    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    all_results = []

    for seed in range(5):  # 你可以改成 30
        res = run(seed)
        all_results.append(res)

    np.save(f"baseline_vtrace_{date}.npy", all_results)
    print("Saved:", f"baseline_vtrace_{date}.npy")