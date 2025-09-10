#!/usr/bin/env python3
"""
학습된 DRL 모델 테스트 스크립트 - 트레이닝 환경과 동일한 설정
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from drl_framework.random_access import STA, Channel, Simulator
from drl_framework.train import SemiMDPLearner

def create_environment(num_slots=200):
    """시뮬레이션 환경 생성 - 트레이닝 환경과 동일하게 설정"""
    # 채널 설정 - 트레이닝과 동일
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.01, obss_duration_range=(10, 150))  # OBSS가 발생하는 채널
    ]
    
    # STA 설정 - 트레이닝과 동일하게 각 채널에 10개씩
    stas = []
    
    # Channel 1의 STA들 (NPCA 지원, 학습 대상) - 10개
    for i in range(10):
        sta = STA(
            sta_id=i,
            channel_id=1,  # OBSS가 발생하는 채널
            primary_channel=channels[1],  # Channel 1이 이들의 primary
            npca_channel=channels[0],     # Channel 0을 NPCA로 사용
            npca_enabled=True,
            ppdu_duration=33,
            radio_transition_time=1
        )
        stas.append(sta)
    
    # Channel 0의 STA들 (기존 방식, 비교용) - 10개
    for i in range(10, 20):
        sta = STA(
            sta_id=i,
            channel_id=0,  # OBSS가 없는 채널
            primary_channel=channels[0],
            npca_channel=None,
            npca_enabled=False,
            ppdu_duration=33,
            radio_transition_time=1
        )
        stas.append(sta)
    
    return Simulator(
        channels=channels,
        stas=stas,
        num_slots=num_slots
    )

def load_trained_model(model_path, device):
    """학습된 모델 로드"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # SemiMDPLearner 초기화
    n_observations = 4  # 상태 차원
    n_actions = 2       # 행동 차원
    learner = SemiMDPLearner(n_observations, n_actions, device)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    learner.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    learner.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    learner.steps_done = checkpoint['steps_done']
    
    # 평가 모드로 설정
    learner.policy_net.eval()
    learner.target_net.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Training steps completed: {learner.steps_done}")
    
    return learner

def run_drl_test(learner, num_episodes=50, num_slots=200):
    """DRL 기법 테스트"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots)
        
        # NPCA 지원 STA들에 학습된 모델 할당
        for sta in simulator.stas:
            if sta.npca_enabled:
                sta.learner = learner
        
        # 시뮬레이션 실행
        simulator.run()
        
        # NPCA 지원 STA들의 성공 전송 슬롯 수만 계산 (학습 대상)
        npca_stas_reward = sum(sta.episode_reward for sta in simulator.stas if sta.npca_enabled)
        episode_rewards.append(npca_stas_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode}: DRL NPCA STAs Successful Slots = {npca_stas_reward:.1f}")
    
    return episode_rewards

def run_baseline_test(strategy, num_episodes=50, num_slots=200):
    """베이스라인 전략 테스트"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots)
        
        # 고정 전략 설정 - NPCA 지원 STA들만 (Channel 1의 STA들)
        for sta in simulator.stas:
            if sta.npca_enabled:  # NPCA 지원 STA만 전략 적용
                if strategy == "always_primary":
                    sta._fixed_action = 0  
                elif strategy == "always_npca":
                    sta._fixed_action = 1  
        
        # 시뮬레이션 실행
        simulator.run()
        
        # NPCA 지원 STA들의 성공 전송 슬롯 수만 계산 (학습 대상)
        npca_stas_reward = sum(sta.episode_reward for sta in simulator.stas if sta.npca_enabled)
        episode_rewards.append(npca_stas_reward)
    
    return episode_rewards

def plot_comparison_results(drl_rewards, primary_rewards, npca_rewards, save_dir="./test_results"):
    """결과 비교 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 에피소드별 보상 비교
    episodes = range(len(drl_rewards))
    ax1.plot(episodes, drl_rewards, label='DRL Policy', alpha=0.8, color='blue', linewidth=2)
    ax1.plot(episodes, primary_rewards, label='Always Primary', alpha=0.7, color='red')
    ax1.plot(episodes, npca_rewards, label='Always NPCA', alpha=0.7, color='green')
    ax1.set_title('Episode Rewards: Successful Transmission Slots')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Successful Transmission Slots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 이동 평균
    if len(drl_rewards) >= 10:
        window = 10
        drl_ma = np.convolve(drl_rewards, np.ones(window)/window, mode='valid')
        primary_ma = np.convolve(primary_rewards, np.ones(window)/window, mode='valid')
        npca_ma = np.convolve(npca_rewards, np.ones(window)/window, mode='valid')
        episodes_ma = range(window-1, len(drl_rewards))
        
        ax2.plot(episodes_ma, drl_ma, label='DRL Policy', color='blue', linewidth=3)
        ax2.plot(episodes_ma, primary_ma, label='Always Primary', color='red', linewidth=2)
        ax2.plot(episodes_ma, npca_ma, label='Always NPCA', color='green', linewidth=2)
    
    ax2.set_title('Moving Average (Window=10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Successful Transmission Slots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 분포 비교
    data = [drl_rewards, primary_rewards, npca_rewards]
    labels = ['DRL Policy', 'Always Primary', 'Always NPCA']
    colors = ['blue', 'red', 'green']
    
    box_plot = ax3.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_title('Reward Distribution Comparison')
    ax3.set_ylabel('Successful Transmission Slots')
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 성능 비교
    means = [np.mean(drl_rewards), np.mean(primary_rewards), np.mean(npca_rewards)]
    stds = [np.std(drl_rewards), np.std(primary_rewards), np.std(npca_rewards)]
    
    bars = ax4.bar(labels, means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.7)
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
    plt.savefig(f"{save_dir}/drl_vs_baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=" * 60)
    print("DRL vs Baseline 전략 비교 테스트")
    print("환경: 트레이닝과 동일한 설정")
    print("=" * 60)
    
    # 설정
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 50
    num_slots = 200
    results_dir = "./test_results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"테스트 에피소드 수: {num_episodes}")
    print(f"에피소드당 슬롯 수: {num_slots}")
    print(f"NPCA enabled STA 수: 10 (Channel 1)")
    print(f"Non-NPCA STA 수: 10 (Channel 0)")
    print(f"OBSS 생성률: 0.01 (Channel 1)")
    print(f"OBSS 지속시간: 10-150 slots")
    print()
    
    try:
        # 학습된 모델 로드
        print("학습된 DRL 모델 로드 중...")
        learner = load_trained_model(model_path, device)
        
        # DRL 정책 테스트
        print("\n1. DRL 정책 테스트 중...")
        drl_rewards = run_drl_test(learner, num_episodes, num_slots)
        
        # 베이스라인 테스트
        print("\n2. Always Primary 전략 테스트 중...")
        primary_rewards = run_baseline_test("always_primary", num_episodes, num_slots)
        
        print("\n3. Always NPCA 전략 테스트 중...")
        npca_rewards = run_baseline_test("always_npca", num_episodes, num_slots)
        
        # 결과 저장
        results_df = pd.DataFrame({
            'drl_policy': drl_rewards,
            'always_primary': primary_rewards,
            'always_npca': npca_rewards
        })
        results_df.to_csv(f"{results_dir}/drl_vs_baseline_comparison.csv", index=False)
        
        # 시각화
        plot_comparison_results(drl_rewards, primary_rewards, npca_rewards, results_dir)
        
        # 통계 출력
        print("\n" + "=" * 60)
        print("결과 요약")
        print("=" * 60)
        
        drl_array = np.array(drl_rewards)
        primary_array = np.array(primary_rewards)
        npca_array = np.array(npca_rewards)
        
        print(f"DRL Policy      - Mean: {drl_array.mean():.2f}, "
              f"Std: {drl_array.std():.2f}, "
              f"Max: {drl_array.max():.1f}, "
              f"Min: {drl_array.min():.1f}")
        
        print(f"Always Primary  - Mean: {primary_array.mean():.2f}, "
              f"Std: {primary_array.std():.2f}, "
              f"Max: {primary_array.max():.1f}, "
              f"Min: {primary_array.min():.1f}")
        
        print(f"Always NPCA     - Mean: {npca_array.mean():.2f}, "
              f"Std: {npca_array.std():.2f}, "
              f"Max: {npca_array.max():.1f}, "
              f"Min: {npca_array.min():.1f}")
        
        # 성능 개선율 계산
        if primary_array.mean() > 0:
            drl_vs_primary = ((drl_array.mean() - primary_array.mean()) / primary_array.mean() * 100)
            drl_vs_npca = ((drl_array.mean() - npca_array.mean()) / npca_array.mean() * 100)
            
            print(f"\nDRL vs Always Primary: {drl_vs_primary:+.1f}%")
            print(f"DRL vs Always NPCA: {drl_vs_npca:+.1f}%")
        
        print(f"\n결과가 {results_dir}/ 에 저장되었습니다.")
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("먼저 main_semi_mdp_training.py를 실행하여 모델을 학습시켜 주세요.")
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()