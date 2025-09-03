import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

from drl_framework.network import ReplayMemory, DQN
from drl_framework.params import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

def select_action(state, policy_net, env, steps_done, device):
    """Epsilon-greedy action selection"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
    #                               device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch  = torch.stack(batch.state).to(device)                # [B, obs_dim]
    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    R_batch      = torch.tensor(batch.cum_reward, device=device).float()  # 옵션 누적 보상
    tau_batch    = torch.tensor(batch.tau, device=device).float()         # 옵션 길이(슬롯 수)
    done_batch   = torch.tensor(batch.done, device=device).float()         # 1.0 if done else 0.0

    # Q(s_t, a)
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    q_sa = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # # V(s_{t+1})
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # max_a' Q_target(s', a') for non-terminal only
    non_final_mask = (done_batch == 0)
    non_final_next_states = torch.stack(
        [s for s, d in zip(batch.next_state, batch.done) if not d]
    ).to(device)

    next_state_values = torch.zeros(len(state_batch), device=device)
    with torch.no_grad():
        if non_final_next_states.numel() > 0:
            # (DDQN 원하면 여기서 policy_net.argmax로 a' 뽑아 target_net.gather 사용)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Loss
    # criterion = nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    td_target = R_batch + (GAMMA ** tau_batch) * next_state_values * (1.0 - done_batch)
    loss = F.smooth_l1_loss(q_sa, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(env, policy_net, target_net, optimizer, device, num_episodes=50):
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_rewards = []

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(env.flatten_dict_values(state),
                             dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        for t in count():
            action = select_action(state, policy_net, env, steps_done, device)
            steps_done += 1

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            total_reward += reward.item()
            done = terminated or truncated

            if not done:
                next_state = torch.tensor(env.flatten_dict_values(observation),
                                          dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model(policy_net, target_net, memory, optimizer, device)

            # Soft update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                             target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(total_reward)
                print(f"Episode {i_episode}: total reward = {total_reward}")
                break

    return episode_rewards
