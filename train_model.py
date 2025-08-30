import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from drl_framework.custom_env import CustomEnv
from drl_framework.network import DQN
from drl_framework.params import *
from drl_framework.train import train

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 환경 초기화
env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                max_terminals=MAX_TERMINALS,
                max_epoch_size=MAX_EPOCH_SIZE,
                max_queue_size=MAX_QUEUE_SIZE,
                reward_weights=REWARD_WEIGHTS)

# 모델 준비
state, _ = env.reset()
n_observations = len(env.flatten_dict_values(state))
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# 학습 실행
num_episodes = 2000 if torch.cuda.is_available() else 50
rewards = train(env, policy_net, target_net, optimizer, device, num_episodes)

# ===== 시각화 =====
plt.figure()
plt.plot(rewards, label="Episode Reward")
# 이동평균(100에피소드)도 같이 보여주면 좋음
if len(rewards) >= 100:
    import numpy as np
    moving_avg = np.convolve(rewards, np.ones(100)/100, mode="valid")
    plt.plot(range(99, len(rewards)), moving_avg, label="Moving Average (100)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Performance")
plt.legend()
plt.show()
