# https://www.youtube.com/@cartoonsondemand

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, chain
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl_framework.custom_env import *
from drl_framework.dqn import *
from drl_framework.params import *

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)
    
# Make test model
def test_model(model, env, iterations=200, simmode="drl"):
    df = pd.DataFrame()
    rewards = []
    for iter in range(iterations):
        next_state, _ = env.reset()
        reward = 0
        for i in range(MAX_EPOCH_SIZE):
            next_state = env.flatten_dict_values(next_state)
            if simmode == "drl":
                selected_action = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device)).max(0)[1].view(1, 1)
            # selected_action = 1 only if the channel quality is good
            elif simmode == "offload_only":
                selected_action = torch.tensor([1], dtype=torch.int64, device=device)
            elif simmode == "local_only":
                selected_action = torch.tensor([0], dtype=torch.int64, device=device)
            # print(f"selected_action: {selected_action}")
            next_state, reward_inst, terminated, truncated, _ = env.step(selected_action.item())
            # print(f"next_state: {next_state}")
            reward += reward_inst
            df_data = pd.DataFrame(data=[next_state.values()], columns=next_state.keys(), index=[iter])
            df_misc = pd.DataFrame(data=[[i, selected_action.item(), reward_inst, reward]], 
                               columns=['epoch', 'action', 'reward_inst', 'reward'],
                               index=[iter])
            df1 = pd.concat([df_misc, df_data], axis=1)
            df = pd.concat([df, df1], axis=0)
        rewards.append(reward)
            
    return df, rewards
    
# Define test env and test model    
test_env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                    max_terminals=MAX_TERMINALS,
                    max_epoch_size=MAX_EPOCH_SIZE,
                    max_queue_size=MAX_QUEUE_SIZE,
                    reward_weights=REWARD_WEIGHTS)
policy_net = torch.load("policy_model.pt", map_location=torch.device('mps'))
policy_net.eval()
print("policy_net: ", policy_net)

for i, simmode in enumerate(["drl", "offload_only", "local_only"]):
# for i, simmode in enumerate(["local_only"]):
    df, rewards = test_model(policy_net, test_env, iterations=200, simmode=simmode)
    filename = simmode + "_test_log.csv"
    df.to_csv(filename)

    # Plot rewards
    plt.figure()
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.plot(rewards_t.numpy())

    means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(19), means))
    plt.plot(means.numpy())
    # Save plot into files
    filename = simmode + "_test_rewards.png"
    plt.savefig(filename)

plt.show()