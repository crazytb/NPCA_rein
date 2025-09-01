import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import count

from drl_framework.network import ReplayMemory, DQN, Transition
from drl_framework.params import *

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

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

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
