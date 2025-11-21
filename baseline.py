"""
A2C + GAE baseline
Ensure every trajectory has the same length.
"""
import gymnasium as gym
import numpy as np
import random
import time
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from models import ActorCritic

import matplotlib.pyplot as plt

ITERATION_NUMS = 1000
SAMPLE_NUMS = 50
LR = 0.01
# Best lambda value is lower than gamma,
# empirically lambda introduces far less bias than gamma for a reasonably accruate value function
GAMMA = 0.99
LAMBDA = 0.98
CLIP_GRAD_NORM = 40


def run(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    gym_name = "CartPole-v1"
    task = gym.make(gym_name)
    task.reset(seed=random_seed)

    discrete = isinstance(task.action_space, gym.spaces.Discrete)
    STATE_DIM = task.observation_space.shape[0]
    ACTION_DIM = task.action_space.n if discrete else task.action_space.shape[0]

    agent = ActorCritic(STATE_DIM, ACTION_DIM)
    optim = torch.optim.Adam(agent.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.0, total_iters=ITERATION_NUMS)

    gamma = GAMMA

    iterations = []
    test_results = []

    init_state, info = task.reset()
    for i in range(ITERATION_NUMS):
        states, actions, advs, v_targets, current_state = roll_out(agent, task, SAMPLE_NUMS, init_state, gamma)
        init_state = current_state
        train(agent, optim, scheduler, states, actions, advs, v_targets, ACTION_DIM)

        # testing
        if (i + 1) % (ITERATION_NUMS // 10) == 0:
            result = test(gym_name, agent)
            print("iteration:", i + 1, "test result:", result / 10.0)
            iterations.append(i + 1)
            test_results.append(result / 10)

    return test_results


def roll_out(agent, task, sample_nums, init_state, gamma):
    states = []
    actions = []
    advs = []
    v_targets = []

    rewards = []
    v_t_s = []
    v_t1_s = []
    dones = []

    state = init_state

    for i in range(sample_nums):
        states.append(state)
        act, v_t = choose_action(agent, state)
        actions.append(act)

        next_state, reward, terminated, truncated, info = task.step(act.numpy())
        done = terminated or truncated

        with torch.no_grad():
            _, v_t1 = agent(torch.Tensor(next_state))

        rewards.append(reward)
        v_t = v_t.detach().numpy()
        v_t1 = v_t1.detach().numpy()
        v_t_s.append(v_t)
        v_t1_s.append(v_t1)
        dones.append(0 if done else 1)

        state = next_state

        if done:
            state, info = task.reset()
            adv, v_target = gae_calculater(rewards, v_t_s, v_t1_s, dones, gamma, LAMBDA)

            advs.append(adv)
            v_targets.append(v_target)
            rewards, v_t_s, v_t1_s, dones = [], [], [], []

    adv, v_target = gae_calculater(rewards, v_t_s, v_t1_s, dones, gamma, LAMBDA)
    advs.append(adv)
    v_targets.append(v_target)

    advs = [item for sublist in advs for item in sublist]
    v_targets = [item for sublist in v_targets for item in sublist]

    return states, actions, advs, v_targets, state

def gae_calculater(rewards, v_t_s, v_t1_s, dones, gamma, lambda_):
    """
    Calculate advantages and target v-values
    """
    batch_size = len(rewards)
    advs = np.zeros(batch_size + 1)
    for t in reversed(range(0, batch_size)):
        delta = rewards[t] - v_t_s[t] + (gamma * v_t1_s[t] * dones[t])
        advs[t] = delta + (gamma * lambda_ * advs[t+1] * dones[t])
    value_target = advs[:batch_size] + np.squeeze(v_t_s)  # target v is calculated from adv.

    return advs[:batch_size], value_target


def train(agent, optim, scheduler, states, actions, advs, v_targets, action_dim):
    states = torch.Tensor(np.array(states))
    actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)

    v_targets = torch.Tensor(v_targets).detach()
    advs = torch.Tensor(advs).detach()

    logits, v = agent(states)
    logits = logits.view(-1, action_dim)
    v = v.view(-1)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    log_probs_act = log_probs.gather(1, actions).view(-1)

    loss_policy = - (advs * log_probs_act).mean()
    loss_critic = F.mse_loss(v_targets, v, reduction='mean')
    loss_entropy = - (log_probs * probs).mean()

    loss = loss_policy + .25 * loss_critic - .001 * loss_entropy
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), CLIP_GRAD_NORM)
    optim.step()
    scheduler.step()


def test(gym_name, agent):
    result = 0
    test_task = gym.make(gym_name)

    for test_epi in range(10):
        state, info = test_task.reset()

        for test_step in range(500):
            act, _ = choose_action(agent, state)

            next_state, reward, terminated, truncated, info = test_task.step(act.numpy())
            done = terminated or truncated

            result += reward
            state = next_state

            if done:
                break

    return result


@torch.no_grad()
def choose_action(agent, state):
    logits, v = agent(torch.Tensor(state))
    act_probs = F.softmax(logits, dim=-1)
    m = Categorical(act_probs)
    act = m.sample()

    return act, v


if __name__ == '__main__':
    date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    total_test_results = []
    for random_seed in range(30):
        test_results = run(random_seed)
        total_test_results.append(test_results)

    dir = '/gemini/code/project/' + date + '.npy'
    np.save(dir, total_test_results)