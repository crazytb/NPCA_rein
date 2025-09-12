#!/usr/bin/env python3
"""
train_model.py 스타일로 Semi-MDP NPCA 환경을 학습하는 스크립트
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN, ReplayMemory
from drl_framework.params import *

def select_action(state, policy_net, device, steps_done):
    """액션 선택 (epsilon-greedy)"""
    sample = torch.rand(1).item()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        torch.exp(torch.tensor(-1. * steps_done / EPS_DECAY)).item()
    
    if sample > eps_threshold:
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return policy_net(state).max(1)[1].item()
    else:
        return torch.randint(0, 2, (1,)).item()

def optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=32):
    """모델 최적화"""
    if len(memory) < batch_size:
        return 0.0
    
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    
    state_batch = torch.stack(batch[0]).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.long).to(device)
    next_state_batch = torch.stack([s for s in batch[2] if s is not None]).to(device)
    reward_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
    tau_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch[5], dtype=torch.bool).to(device)
    
    # Q(s,a) 계산
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # V(s') 계산 - 종료되지 않은 상태만
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        if len(next_state_batch) > 0:
            next_state_values[~done_batch] = target_net(next_state_batch).max(1)[0]
    
    # Semi-MDP 할인률 적용: gamma^tau
    gamma_tau = GAMMA ** tau_batch
    expected_state_action_values = (next_state_values * gamma_tau) + reward_batch
    
    # Huber Loss 계산
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)
    
    # 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_semi_mdp(env, policy_net, target_net, optimizer, device, num_episodes=1000):
    """Semi-MDP 환경 학습"""
    memory = ReplayMemory(10000)
    episode_rewards = []
    episode_losses = []
    steps_done = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        done = False
        while not done:
            # 액션 선택
            action = select_action(obs_tensor, policy_net, device, steps_done)
            steps_done += 1
            
            # 환경 스텝
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # 다음 상태
            next_obs_tensor = None if done else torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            # 메모리에 저장 (Semi-MDP: tau는 info에서 가져오기)
            tau = info.get('duration', 1)  # 옵션 지속 시간
            memory.push(obs_tensor, action, next_obs_tensor, reward, tau, done)
            
            # 상태 업데이트
            obs_tensor = next_obs_tensor
            
            # 모델 최적화
            loss = optimize_model(memory, policy_net, target_net, optimizer, device)
            episode_loss += loss
            step_count += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss / step_count if step_count > 0 else 0)
        
        # Target 네트워크 업데이트
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 진행상황 출력
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            avg_loss = np.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else episode_loss
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Avg Loss = {avg_loss:.4f}, Epsilon = {eps:.3f}")
    
    return episode_rewards, episode_losses

def main():
    """메인 함수"""
    print("=" * 60)
    print("Semi-MDP NPCA 환경 학습 (train_model.py 스타일)")
    print("=" * 60)
    
    # Device 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    
    # 환경 초기화
    env = NPCASemiMDPEnv(num_stas=10, num_slots=200)
    
    # 모델 준비
    obs, _ = env.reset()
    n_observations = len(obs)
    n_actions = 2  # Stay Primary (0) or Go NPCA (1)
    
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    
    # 학습 실행
    num_episodes = 1000 if device != torch.device("cpu") else 100
    print(f"학습 에피소드 수: {num_episodes}")
    
    rewards, losses = train_semi_mdp(env, policy_net, target_net, optimizer, device, num_episodes)
    
    # 모델 저장
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': rewards,
        'episode_losses': losses,
    }, './semi_mdp_model_trainpy_style.pth')
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 보상 그래프
    ax1.plot(rewards, label="Episode Reward", alpha=0.7)
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode="valid")
        ax1.plot(range(49, len(rewards)), moving_avg, label="Moving Average (50)", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Performance - Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 손실 그래프
    ax2.plot(losses, label="Episode Loss", alpha=0.7, color='red')
    if len(losses) >= 50:
        moving_avg_loss = np.convolve(losses, np.ones(50)/50, mode="valid")
        ax2.plot(range(49, len(losses)), moving_avg_loss, label="Moving Average (50)", linewidth=2, color='darkred')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Loss")
    ax2.set_title("Training Performance - Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./semi_mdp_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n학습 완료!")
    print(f"모델 저장됨: ./semi_mdp_model_trainpy_style.pth")
    print(f"결과 그래프: ./semi_mdp_training_results.png")

if __name__ == "__main__":
    main()