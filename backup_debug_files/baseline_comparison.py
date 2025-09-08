#!/usr/bin/env python3
"""
학습된 DRL 정책 vs "Always NPCA" 전략 성능 비교

이 스크립트는 다음 세 가지 전략의 성능을 비교합니다:
1. 학습된 DRL 정책 (Semi-MDP 기반)
2. Always NPCA (항상 GoNPCA 선택)
3. Always StayPrimary (항상 StayPrimary 선택)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel
from drl_framework.train import SemiMDPLearner
from drl_framework.network import DQN
from drl_framework.params import *

def create_test_config():
    """테스트를 위한 설정 생성"""
    # 채널 설정 (훈련과 동일)
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))  # NPCA channel
    ]
    
    # STA 설정 - Channel 1의 NPCA 지원 STA들만 테스트 (10개)
    stas_config = []
    for i in range(10):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    return channels, stas_config

def run_baseline_test(channels, stas_config, strategy="always_npca", num_episodes=100, num_slots_per_episode=200, device="cpu"):
    """베이스라인 전략으로 테스트 실행"""
    from drl_framework.random_access import Simulator
    
    # 시뮬레이터 생성 (학습 모드 비활성화)
    simulator = Simulator(channels, stas_config, device=device, learning_mode=False)
    
    episode_rewards = []
    decision_log = []
    
    print(f"\n{strategy} 전략 테스트 시작...")
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"Episode {episode}/{num_episodes}")
            
        # 에피소드 초기화
        simulator.reset()
        episode_reward = 0.0
        
        for slot in range(num_slots_per_episode):
            # 각 STA에 대해 결정이 필요한 경우 고정 전략 적용
            for sta in simulator.stas:
                if hasattr(sta, '_needs_decision') and sta._needs_decision:
                    # 고정 전략에 따라 액션 선택
                    if strategy == "always_npca":
                        action = 1  # GoNPCA
                    elif strategy == "always_primary":
                        action = 0  # StayPrimary
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    # 현재 상태 관찰
                    obs_dict = sta.get_observation()
                    
                    # 결정 로그 기록
                    decision_log.append({
                        'episode': episode,
                        'slot': slot,
                        'sta_id': sta.sta_id,
                        'primary_channel_obss_occupied_remained': obs_dict.get('primary_channel_obss_occupied_remained', 0),
                        'radio_transition_time': obs_dict.get('radio_transition_time', 0),
                        'tx_duration': obs_dict.get('tx_duration', 0),
                        'cw_index': obs_dict.get('cw_index', 0),
                        'action': action,
                        'strategy': strategy
                    })
                    
                    # 액션 실행
                    sta._select_action_fixed(action)
                    sta._needs_decision = False
            
            # 시뮬레이션 한 스텝 진행
            simulator.step(slot)
            
        # 에피소드 종료 후 보상 수집
        total_reward = sum(sta.episode_reward for sta in simulator.stas)
        episode_rewards.append(total_reward / 100)  # 정규화
        
        # STA 보상 초기화
        for sta in simulator.stas:
            sta.episode_reward = 0.0
    
    return episode_rewards, decision_log

def run_drl_test(channels, stas_config, model_path, num_episodes=100, num_slots_per_episode=200, device="cpu"):
    """학습된 DRL 모델로 테스트 실행"""
    from drl_framework.random_access import Simulator
    
    # 학습된 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # DQN 네트워크 생성
    state_size = 4  # (primary_channel_obss_occupied_remained, radio_transition_time, tx_duration, cw_index)
    action_size = 2  # (StayPrimary, GoNPCA)
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()
    
    # 시뮬레이터 생성 (학습 모드 활성화)
    simulator = Simulator(channels, stas_config, device=device, learning_mode=True)
    
    # 학습된 모델을 STA들에 할당
    for sta in simulator.stas:
        if hasattr(sta, 'learner'):
            sta.learner.policy_net = policy_net
            sta.learner.steps_done = checkpoint['steps_done']
    
    episode_rewards = []
    decision_log = []
    
    print(f"\n학습된 DRL 정책 테스트 시작...")
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"Episode {episode}/{num_episodes}")
            
        # 에피소드 초기화
        simulator.reset()
        
        # CSV 로깅을 위한 설정
        for sta in simulator.stas:
            sta.decision_log = decision_log
            sta.current_episode = episode
        
        for slot in range(num_slots_per_episode):
            simulator.step(slot)
            
        # 에피소드 종료 후 보상 수집
        total_reward = sum(sta.episode_reward for sta in simulator.stas)
        episode_rewards.append(total_reward / 100)  # 정규화
        
        # STA 보상 초기화
        for sta in simulator.stas:
            sta.episode_reward = 0.0
    
    return episode_rewards, decision_log

def plot_comparison_results(drl_rewards, npca_rewards, primary_rewards, save_dir="./comparison_results"):
    """비교 결과를 플롯으로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 1. 에피소드별 보상 비교
    plt.subplot(2, 2, 1)
    episodes = range(len(drl_rewards))
    plt.plot(episodes, drl_rewards, label='DRL Policy', alpha=0.7, color='blue')
    plt.plot(episodes, npca_rewards, label='Always NPCA', alpha=0.7, color='red')
    plt.plot(episodes, primary_rewards, label='Always Primary', alpha=0.7, color='green')
    plt.title('Episode Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Running Average 비교
    plt.subplot(2, 2, 2)
    window = 20
    
    def running_avg(data, window):
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    drl_avg = running_avg(drl_rewards, window)
    npca_avg = running_avg(npca_rewards, window)
    primary_avg = running_avg(primary_rewards, window)
    
    plt.plot(episodes, drl_avg, label=f'DRL Policy (avg)', linewidth=2, color='blue')
    plt.plot(episodes, npca_avg, label=f'Always NPCA (avg)', linewidth=2, color='red')
    plt.plot(episodes, primary_avg, label=f'Always Primary (avg)', linewidth=2, color='green')
    plt.title(f'Running Average Comparison (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 보상 분포 박스플롯
    plt.subplot(2, 2, 3)
    data = [drl_rewards, npca_rewards, primary_rewards]
    labels = ['DRL Policy', 'Always NPCA', 'Always Primary']
    plt.boxplot(data, labels=labels)
    plt.title('Reward Distribution')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # 4. 누적 보상 비교
    plt.subplot(2, 2, 4)
    drl_cum = np.cumsum(drl_rewards)
    npca_cum = np.cumsum(npca_rewards)
    primary_cum = np.cumsum(primary_rewards)
    
    plt.plot(episodes, drl_cum, label='DRL Policy', linewidth=2, color='blue')
    plt.plot(episodes, npca_cum, label='Always NPCA', linewidth=2, color='red')
    plt.plot(episodes, primary_cum, label='Always Primary', linewidth=2, color='green')
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to {save_dir}/strategy_comparison.png")

def main():
    """메인 비교 함수"""
    print("="*60)
    print("DRL vs Baseline 전략 성능 비교")
    print("="*60)
    
    # 설정 생성
    channels, stas_config = create_test_config()
    
    # 테스트 파라미터
    num_episodes = 100
    num_slots_per_episode = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    
    print(f"Device: {device}")
    print(f"Test episodes: {num_episodes}")
    print(f"Slots per episode: {num_slots_per_episode}")
    print(f"Model path: {model_path}")
    print()
    
    # 1. 학습된 DRL 정책 테스트
    if os.path.exists(model_path):
        drl_rewards, drl_log = run_drl_test(channels, stas_config, model_path, num_episodes, num_slots_per_episode, device)
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    
    # 2. Always NPCA 전략 테스트  
    npca_rewards, npca_log = run_baseline_test(channels, stas_config, "always_npca", num_episodes, num_slots_per_episode, device)
    
    # 3. Always StayPrimary 전략 테스트
    primary_rewards, primary_log = run_baseline_test(channels, stas_config, "always_primary", num_episodes, num_slots_per_episode, device)
    
    # 결과 저장
    results_dir = "./comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # CSV 파일 저장
    pd.DataFrame({'drl': drl_rewards, 'always_npca': npca_rewards, 'always_primary': primary_rewards}).to_csv(f"{results_dir}/rewards_comparison.csv", index=False)
    
    if drl_log:
        pd.DataFrame(drl_log).to_csv(f"{results_dir}/drl_decisions.csv", index=False)
    if npca_log:
        pd.DataFrame(npca_log).to_csv(f"{results_dir}/npca_decisions.csv", index=False)
    if primary_log:
        pd.DataFrame(primary_log).to_csv(f"{results_dir}/primary_decisions.csv", index=False)
    
    # 성능 비교 플롯
    plot_comparison_results(drl_rewards, npca_rewards, primary_rewards, results_dir)
    
    # 통계 결과 출력
    print("\n" + "="*60)
    print("성능 비교 결과")
    print("="*60)
    print(f"DRL Policy      - Mean: {np.mean(drl_rewards):.3f}, Std: {np.std(drl_rewards):.3f}, Max: {np.max(drl_rewards):.3f}")
    print(f"Always NPCA     - Mean: {np.mean(npca_rewards):.3f}, Std: {np.std(npca_rewards):.3f}, Max: {np.max(npca_rewards):.3f}")  
    print(f"Always Primary  - Mean: {np.mean(primary_rewards):.3f}, Std: {np.std(primary_rewards):.3f}, Max: {np.max(primary_rewards):.3f}")
    print()
    print(f"DRL vs Always NPCA    - Improvement: {((np.mean(drl_rewards) - np.mean(npca_rewards)) / np.mean(npca_rewards) * 100):+.1f}%")
    print(f"DRL vs Always Primary - Improvement: {((np.mean(drl_rewards) - np.mean(primary_rewards)) / np.mean(primary_rewards) * 100):+.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()