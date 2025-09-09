#!/usr/bin/env python3
"""
단순한 베이스라인 비교 테스트 - DQN 학습된 모델 없이 고정 전략만 비교
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from drl_framework.random_access import STA, Channel, Simulator

def create_environment(num_slots=200, num_stas=10):
    """시뮬레이션 환경 생성"""
    # Primary 채널: OBSS 발생
    primary_channel = Channel(
        channel_id=0,
        obss_generation_rate=0.3,  
        obss_duration_range=(20, 40)
    )
    
    # NPCA 채널: OBSS 없음
    npca_channel = Channel(
        channel_id=1,
        obss_generation_rate=0.0,  
        obss_duration_range=(0, 0)
    )
    
    channels = [primary_channel, npca_channel]
    
    # STA 설정
    stas = []
    for i in range(num_stas):
        sta = STA(
            sta_id=i,
            channel_id=0,
            primary_channel=primary_channel,
            npca_channel=npca_channel,
            npca_enabled=True
        )
        stas.append(sta)
    
    return Simulator(
        channels=channels,
        stas=stas,
        num_slots=num_slots
    )

def run_baseline_test(strategy, num_episodes=50, num_slots=200, num_stas=10):
    """베이스라인 전략 테스트"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots, num_stas)
        
        # 고정 전략 설정
        for sta in simulator.stas:
            if strategy == "always_primary":
                sta._fixed_action = 0  
            elif strategy == "always_npca":
                sta._fixed_action = 1  
        
        # 시뮬레이션 실행
        simulator.run()
        
        # 총 성공 전송 슬롯 수 계산
        total_reward = sum(sta.episode_reward for sta in simulator.stas)
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode}: Total Successful Slots = {total_reward:.1f}")
    
    return episode_rewards

def plot_results(primary_rewards, npca_rewards, save_dir="./baseline_results"):
    """결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 에피소드별 보상 비교
    episodes = range(len(primary_rewards))
    ax1.plot(episodes, primary_rewards, label='Always Primary', alpha=0.8, color='red')
    ax1.plot(episodes, npca_rewards, label='Always NPCA', alpha=0.8, color='green')
    ax1.set_title('Episode Rewards: Successful Transmission Slots')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Successful Transmission Slots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 이동 평균
    if len(primary_rewards) >= 10:
        window = 10
        primary_ma = np.convolve(primary_rewards, np.ones(window)/window, mode='valid')
        npca_ma = np.convolve(npca_rewards, np.ones(window)/window, mode='valid')
        episodes_ma = range(window-1, len(primary_rewards))
        
        ax2.plot(episodes_ma, primary_ma, label='Always Primary', color='red', linewidth=2)
        ax2.plot(episodes_ma, npca_ma, label='Always NPCA', color='green', linewidth=2)
    
    ax2.set_title('Moving Average (Window=10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Successful Transmission Slots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 분포
    data = [primary_rewards, npca_rewards]
    labels = ['Always Primary', 'Always NPCA']
    ax3.boxplot(data, labels=labels)
    ax3.set_title('Reward Distribution')
    ax3.set_ylabel('Successful Transmission Slots')
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 성능 비교
    means = [np.mean(primary_rewards), np.mean(npca_rewards)]
    stds = [np.std(primary_rewards), np.std(npca_rewards)]
    
    bars = ax4.bar(labels, means, yerr=stds, capsize=5, 
                   color=['red', 'green'], alpha=0.7)
    ax4.set_title('Average Performance Comparison')
    ax4.set_ylabel('Mean Successful Transmission Slots')
    ax4.grid(True, alpha=0.3)
    
    # 바 위에 수치 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=" * 60)
    print("베이스라인 전략 비교 (새로운 보상 구조)")
    print("보상: 에피소드당 성공적으로 전송한 총 슬롯 수")
    print("=" * 60)
    
    num_episodes = 50
    num_slots = 200
    num_stas = 10
    results_dir = "./baseline_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"테스트 에피소드 수: {num_episodes}")
    print(f"에피소드당 슬롯 수: {num_slots}")
    print(f"NPCA enabled STA 수: {num_stas}")
    print()
    
    # Always Primary 테스트
    print("1. Always Primary 전략 테스트 중...")
    primary_rewards = run_baseline_test("always_primary", num_episodes, num_slots, num_stas)
    
    # Always NPCA 테스트
    print("\n2. Always NPCA 전략 테스트 중...")
    npca_rewards = run_baseline_test("always_npca", num_episodes, num_slots, num_stas)
    
    # 결과 저장
    results_df = pd.DataFrame({
        'always_primary': primary_rewards,
        'always_npca': npca_rewards
    })
    results_df.to_csv(f"{results_dir}/baseline_comparison.csv", index=False)
    
    # 시각화
    plot_results(primary_rewards, npca_rewards, results_dir)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    
    primary_array = np.array(primary_rewards)
    npca_array = np.array(npca_rewards)
    
    print(f"Always Primary  - Mean: {primary_array.mean():.2f}, "
          f"Std: {primary_array.std():.2f}, "
          f"Max: {primary_array.max():.1f}, "
          f"Min: {primary_array.min():.1f}")
    
    print(f"Always NPCA     - Mean: {npca_array.mean():.2f}, "
          f"Std: {npca_array.std():.2f}, "
          f"Max: {npca_array.max():.1f}, "
          f"Min: {npca_array.min():.1f}")
    
    # 성능 개선율
    if primary_array.mean() > 0:
        improvement = ((npca_array.mean() - primary_array.mean()) / primary_array.mean() * 100)
        print(f"\nAlways NPCA vs Always Primary: {improvement:+.1f}%")
    
    print(f"\n결과가 {results_dir}/ 에 저장되었습니다.")

if __name__ == "__main__":
    main()