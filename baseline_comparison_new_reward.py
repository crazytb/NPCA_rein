#!/usr/bin/env python3
"""
새로운 보상 구조(성공 전송 슬롯 수)로 DRL 정책과 베이스라인 비교
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from drl_framework.random_access import STA, Channel, Simulator
from drl_framework.network import DQN
def create_environment(num_slots=200, num_stas=10):
    """시뮬레이션 환경 생성"""
    # 채널 설정
    primary_channel = Channel(
        channel_id=0,
        obss_generation_rate=0.3,  # Primary 채널에 OBSS 발생
        obss_duration_range=(20, 40)
    )
    
    npca_channel = Channel(
        channel_id=1,
        obss_generation_rate=0.0,  # NPCA 채널에는 OBSS 없음
        obss_duration_range=(0, 0)
    )
    
    channels = [primary_channel, npca_channel]
    
    # STA 설정 - NPCA enabled STAs만 생성
    stas = []
    for i in range(num_stas):
        sta = STA(
            sta_id=i,
            channel_id=0,  # Primary 채널 ID
            primary_channel=primary_channel,
            npca_channel=npca_channel,
            npca_enabled=True
        )
        stas.append(sta)
    
    # 시뮬레이터 생성
    simulator = Simulator(
        channels=channels,
        stas=stas,
        num_slots=num_slots
    )
    
    return simulator

def run_drl_test(model_path, num_episodes=100, num_slots=200, num_stas=10):
    """DRL 정책 테스트"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 모델 로드
    model = DQN(n_observations=4, n_actions=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'policy_net_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['policy_net_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    episode_rewards = []
    decision_log = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots, num_stas)
        
        # DRL 정책을 사용할 STA 설정
        for sta in simulator.stas:
            if hasattr(sta, 'npca_enabled') and sta.npca_enabled:
                class MockLearner:
                    def __init__(self, policy_net, device):
                        self.policy_net = policy_net
                        self.device = device
                        self.memory = None
                        self.steps_done = 0
                    
                    def select_action(self, state_tensor, training=False):
                        with torch.no_grad():
                            if state_tensor.dim() == 1:
                                state_tensor = state_tensor.unsqueeze(0)
                            q_values = self.policy_net(state_tensor)
                            return q_values.max(1)[1].item()
                
                sta.learner = MockLearner(model, device)
        
        # 에피소드 실행
        simulator.run()
        
        # 보상 계산 (모든 NPCA enabled STA의 성공 전송 슬롯 수 합)
        total_reward = sum(sta.episode_reward for sta in simulator.stas 
                          if hasattr(sta, 'npca_enabled') and sta.npca_enabled)
        
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode}: Total Reward = {total_reward:.1f}")
    
    return episode_rewards, decision_log

def run_baseline_test(strategy, num_episodes=100, num_slots=200, num_stas=10):
    """베이스라인 전략 테스트"""
    episode_rewards = []
    decision_log = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots, num_stas)
        
        # 고정 전략 설정
        for sta in simulator.stas:
            if hasattr(sta, 'npca_enabled') and sta.npca_enabled:
                if strategy == "always_primary":
                    sta._fixed_action = 0  # Always stay primary
                elif strategy == "always_npca":
                    sta._fixed_action = 1  # Always go NPCA
        
        # 에피소드 실행
        simulator.run()
        
        # 보상 계산 (모든 NPCA enabled STA의 성공 전송 슬롯 수 합)
        total_reward = sum(sta.episode_reward for sta in simulator.stas 
                          if hasattr(sta, 'npca_enabled') and sta.npca_enabled)
        
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode}: Total Reward = {total_reward:.1f}")
    
    return episode_rewards, decision_log

def plot_comparison(drl_rewards, primary_rewards, npca_rewards, save_dir="./comparison_results"):
    """결과 비교 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 에피소드별 보상 비교
    episodes = range(len(drl_rewards))
    ax1.plot(episodes, drl_rewards, label='DRL Policy', alpha=0.7, color='blue')
    ax1.plot(episodes, primary_rewards, label='Always Primary', alpha=0.7, color='red')
    ax1.plot(episodes, npca_rewards, label='Always NPCA', alpha=0.7, color='green')
    ax1.set_title('Episode Rewards Comparison (New Reward Structure)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Successful Transmission Slots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 이동 평균 (윈도우 크기: 20)
    def moving_average(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    if len(drl_rewards) >= 20:
        drl_ma = moving_average(drl_rewards, 20)
        primary_ma = moving_average(primary_rewards, 20)
        npca_ma = moving_average(npca_rewards, 20)
        episodes_ma = range(19, len(drl_rewards))
        
        ax2.plot(episodes_ma, drl_ma, label='DRL Policy', color='blue', linewidth=2)
        ax2.plot(episodes_ma, primary_ma, label='Always Primary', color='red', linewidth=2)
        ax2.plot(episodes_ma, npca_ma, label='Always NPCA', color='green', linewidth=2)
    
    ax2.set_title('Moving Average (Window=20)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Successful Transmission Slots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 분포 박스 플롯
    data = [drl_rewards, primary_rewards, npca_rewards]
    labels = ['DRL Policy', 'Always Primary', 'Always NPCA']
    ax3.boxplot(data, labels=labels)
    ax3.set_title('Reward Distribution')
    ax3.set_ylabel('Successful Transmission Slots')
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 성능 바 그래프
    means = [np.mean(drl_rewards), np.mean(primary_rewards), np.mean(npca_rewards)]
    stds = [np.std(drl_rewards), np.std(primary_rewards), np.std(npca_rewards)]
    
    bars = ax4.bar(labels, means, yerr=stds, capsize=5, 
                   color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_title('Average Performance Comparison')
    ax4.set_ylabel('Mean Successful Transmission Slots')
    ax4.grid(True, alpha=0.3)
    
    # 바 위에 수치 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/new_reward_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=" * 60)
    print("새로운 보상 구조 기반 베이스라인 비교")
    print("=" * 60)
    
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 50
    num_slots = 200
    num_stas = 10
    results_dir = "./comparison_results_new_reward"
    
    if not os.path.exists(model_path):
        print(f"모델 파일이 없습니다: {model_path}")
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"테스트 에피소드 수: {num_episodes}")
    print(f"에피소드당 슬롯 수: {num_slots}")
    print(f"NPCA enabled STA 수: {num_stas}")
    print()
    
    # DRL 정책 테스트
    print("1. DRL 정책 테스트 중...")
    drl_rewards, drl_log = run_drl_test(model_path, num_episodes, num_slots, num_stas)
    
    # Always Primary 테스트
    print("\n2. Always Primary 전략 테스트 중...")
    primary_rewards, primary_log = run_baseline_test("always_primary", num_episodes, num_slots, num_stas)
    
    # Always NPCA 테스트
    print("\n3. Always NPCA 전략 테스트 중...")
    npca_rewards, npca_log = run_baseline_test("always_npca", num_episodes, num_slots, num_stas)
    
    # 결과 저장
    results_df = pd.DataFrame({
        'drl': drl_rewards,
        'always_primary': primary_rewards,
        'always_npca': npca_rewards
    })
    results_df.to_csv(f"{results_dir}/rewards_comparison_new.csv", index=False)
    
    # 시각화
    plot_comparison(drl_rewards, primary_rewards, npca_rewards, results_dir)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("결과 요약 (성공 전송 슬롯 수 기준)")
    print("=" * 60)
    
    strategies = {
        'DRL Policy': drl_rewards,
        'Always Primary': primary_rewards,
        'Always NPCA': npca_rewards
    }
    
    for name, rewards in strategies.items():
        rewards_array = np.array(rewards)
        print(f"{name:15s} - Mean: {rewards_array.mean():.3f}, "
              f"Std: {rewards_array.std():.3f}, "
              f"Max: {rewards_array.max():.3f}, "
              f"Min: {rewards_array.min():.3f}")
    
    # 성능 개선율 계산
    drl_mean = np.mean(drl_rewards)
    primary_mean = np.mean(primary_rewards)
    npca_mean = np.mean(npca_rewards)
    
    print("\n" + "-" * 60)
    print("성능 개선율:")
    print(f"DRL vs Always Primary : {((drl_mean - primary_mean) / primary_mean * 100):+.1f}%")
    print(f"DRL vs Always NPCA    : {((drl_mean - npca_mean) / npca_mean * 100):+.1f}%")
    
    print(f"\n결과가 {results_dir}/ 에 저장되었습니다.")

if __name__ == "__main__":
    main()