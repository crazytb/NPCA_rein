#!/usr/bin/env python3
"""
최종 DRL vs 베이스라인 성능 비교
새로운 보상 구조 기반으로 DRL 정책, Always Primary, Always NPCA 비교
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

def run_drl_test(model_path, num_episodes=50, num_slots=200, num_stas=10):
    """DRL 정책 테스트 - 시뮬레이터 직접 사용"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 모델 로드
    model = DQN(n_observations=4, n_actions=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots, num_stas)
        
        # DRL 정책을 위한 간소화된 Learner 설정
        class SimpleLearner:
            def __init__(self, policy_net, device):
                self.policy_net = policy_net
                self.device = device
                self.steps_done = 0
                self.memory = None  # 테스트 시에는 메모리 사용 안함
            
            def select_action(self, state_tensor, training=False):
                """Greedy 액션 선택 (테스트 모드)"""
                with torch.no_grad():
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    return q_values.max(1)[1].item()
        
        # 모든 NPCA enabled STA에 DRL 정책 할당
        for sta in simulator.stas:
            if hasattr(sta, 'npca_enabled') and sta.npca_enabled:
                sta.learner = SimpleLearner(model, device)
        
        # 시뮬레이션 실행
        simulator.run()
        
        # 총 성공 전송 슬롯 수 계산
        total_reward = sum(sta.episode_reward for sta in simulator.stas)
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode}: DRL Total Successful Slots = {total_reward:.1f}")
    
    return episode_rewards

def run_baseline_test(strategy, num_episodes=50, num_slots=200, num_stas=10):
    """베이스라인 전략 테스트"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots, num_stas)
        
        # 고정 전략 설정
        for sta in simulator.stas:
            if hasattr(sta, 'npca_enabled') and sta.npca_enabled:
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
            print(f"  Episode {episode}: {strategy} Total Successful Slots = {total_reward:.1f}")
    
    return episode_rewards

def plot_comparison(drl_rewards, primary_rewards, npca_rewards, save_dir="./final_comparison"):
    """최종 결과 비교 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 에피소드별 보상 비교
    episodes = range(len(drl_rewards))
    ax1.plot(episodes, drl_rewards, label='DRL Policy', alpha=0.8, color='blue', linewidth=1.5)
    ax1.plot(episodes, primary_rewards, label='Always Primary', alpha=0.8, color='red', linewidth=1.5)
    ax1.plot(episodes, npca_rewards, label='Always NPCA', alpha=0.8, color='green', linewidth=1.5)
    ax1.set_title('Episode Performance: Successful Transmission Slots', fontsize=14, fontweight='bold')
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
        
        ax2.plot(episodes_ma, drl_ma, label='DRL Policy', color='blue', linewidth=2)
        ax2.plot(episodes_ma, primary_ma, label='Always Primary', color='red', linewidth=2)
        ax2.plot(episodes_ma, npca_ma, label='Always NPCA', color='green', linewidth=2)
    
    ax2.set_title('Moving Average Performance (Window=10)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Successful Transmission Slots')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 분포
    data = [drl_rewards, primary_rewards, npca_rewards]
    labels = ['DRL Policy', 'Always Primary', 'Always NPCA']
    box_plot = ax3.boxplot(data, tick_labels=labels, patch_artist=True)
    
    # 박스플롯 색상 설정
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Performance Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Successful Transmission Slots')
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 성능 바 그래프
    means = [np.mean(drl_rewards), np.mean(primary_rewards), np.mean(npca_rewards)]
    stds = [np.std(drl_rewards), np.std(primary_rewards), np.std(npca_rewards)]
    
    bars = ax4.bar(labels, means, yerr=stds, capsize=5, 
                   color=['blue', 'red', 'green'], alpha=0.7, 
                   edgecolor='black', linewidth=1)
    ax4.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Successful Transmission Slots')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 바 위에 수치 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_drl_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=" * 70)
    print("최종 DRL vs 베이스라인 성능 비교")
    print("새로운 보상 구조: 성공적으로 전송한 총 슬롯 수")
    print("=" * 70)
    
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 50
    num_slots = 200
    num_stas = 10
    results_dir = "./final_comparison"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("main_semi_mdp_training.py를 실행하여 모델을 학습시키세요.")
        return
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📊 테스트 설정:")
    print(f"  - 에피소드 수: {num_episodes}")
    print(f"  - 에피소드당 슬롯 수: {num_slots}")
    print(f"  - NPCA enabled STA 수: {num_stas}")
    print()
    
    # 1. DRL 정책 테스트
    print("🤖 1. DRL 정책 테스트 중...")
    try:
        drl_rewards = run_drl_test(model_path, num_episodes, num_slots, num_stas)
        print(f"   ✅ DRL 테스트 완료!")
    except Exception as e:
        print(f"   ❌ DRL 테스트 실패: {e}")
        return
    
    # 2. Always Primary 테스트
    print("\n🔴 2. Always Primary 전략 테스트 중...")
    primary_rewards = run_baseline_test("always_primary", num_episodes, num_slots, num_stas)
    print(f"   ✅ Always Primary 테스트 완료!")
    
    # 3. Always NPCA 테스트
    print("\n🟢 3. Always NPCA 전략 테스트 중...")
    npca_rewards = run_baseline_test("always_npca", num_episodes, num_slots, num_stas)
    print(f"   ✅ Always NPCA 테스트 완료!")
    
    # 4. 결과 저장
    results_df = pd.DataFrame({
        'DRL_Policy': drl_rewards,
        'Always_Primary': primary_rewards,
        'Always_NPCA': npca_rewards
    })
    results_df.to_csv(f"{results_dir}/final_comparison_results.csv", index=False)
    
    # 5. 시각화
    print("\n📈 4. 결과 시각화 중...")
    plot_comparison(drl_rewards, primary_rewards, npca_rewards, results_dir)
    
    # 6. 통계 분석 및 출력
    print("\n" + "=" * 70)
    print("🏆 최종 성능 분석 결과")
    print("=" * 70)
    
    strategies = {
        'DRL Policy': np.array(drl_rewards),
        'Always Primary': np.array(primary_rewards),
        'Always NPCA': np.array(npca_rewards)
    }
    
    # 기본 통계
    print("📊 기본 통계:")
    for name, rewards in strategies.items():
        print(f"  {name:15s} - Mean: {rewards.mean():6.2f}, "
              f"Std: {rewards.std():5.2f}, "
              f"Max: {rewards.max():6.1f}, "
              f"Min: {rewards.min():6.1f}")
    
    # 성능 비교
    print("\n🚀 성능 개선율:")
    drl_mean = strategies['DRL Policy'].mean()
    primary_mean = strategies['Always Primary'].mean()
    npca_mean = strategies['Always NPCA'].mean()
    
    drl_vs_primary = ((drl_mean - primary_mean) / primary_mean * 100)
    drl_vs_npca = ((drl_mean - npca_mean) / npca_mean * 100)
    
    print(f"  DRL vs Always Primary : {drl_vs_primary:+6.1f}%")
    print(f"  DRL vs Always NPCA    : {drl_vs_npca:+6.1f}%")
    
    # 최고 성능 전략
    best_strategy = max(strategies.items(), key=lambda x: x[1].mean())
    print(f"\n🏅 최고 성능 전략: {best_strategy[0]} (평균 {best_strategy[1].mean():.2f} 슬롯)")
    
    print(f"\n💾 모든 결과가 {results_dir}/ 에 저장되었습니다.")
    print("   - final_comparison_results.csv: 원시 데이터")  
    print("   - final_drl_comparison.png: 시각화 결과")

if __name__ == "__main__":
    main()