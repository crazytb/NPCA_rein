#!/usr/bin/env python3
"""
포괄적 성능 비교 테스트: DRL vs 베이스라인 전략들
트레이닝 환경과 동일한 설정으로 모든 기법 비교
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
        return None
    
    try:
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
        
        return learner
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

def run_strategy_test(strategy, learner=None, num_episodes=50, num_slots=200):
    """특정 전략으로 테스트 실행"""
    episode_rewards = []
    episode_details = []
    
    for episode in range(num_episodes):
        simulator = create_environment(num_slots)
        
        # 전략별 설정
        for sta in simulator.stas:
            if sta.npca_enabled:  # NPCA 지원 STA만 설정
                if strategy == "drl_policy" and learner is not None:
                    sta.learner = learner
                elif strategy == "always_primary":
                    sta._fixed_action = 0  
                elif strategy == "always_npca":
                    sta._fixed_action = 1
                elif strategy == "random":
                    # 매 결정마다 랜덤 선택하도록 설정
                    pass  # 랜덤은 STA 내부에서 처리
        
        # 시뮬레이션 실행
        simulator.run()
        
        # NPCA 지원 STA들의 성과 수집
        npca_stas = [sta for sta in simulator.stas if sta.npca_enabled]
        episode_reward = sum(sta.episode_reward for sta in npca_stas)
        episode_rewards.append(episode_reward)
        
        # 세부 정보 수집
        total_occupancy = sum(sta.channel_occupancy_time for sta in npca_stas)
        episode_details.append({
            'episode': episode,
            'strategy': strategy,
            'total_reward': episode_reward,
            'total_occupancy_slots': total_occupancy,
            'avg_reward_per_sta': episode_reward / len(npca_stas),
            'occupancy_ratio': total_occupancy / (num_slots * len(npca_stas))
        })
    
    return episode_rewards, episode_details

def plot_comprehensive_results(results_dict, details_df, save_dir="./comprehensive_results"):
    """포괄적 결과 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 색상 및 스타일 설정
    colors = {
        'DRL Policy': '#1f77b4',
        'Always Primary': '#ff7f0e', 
        'Always NPCA': '#2ca02c',
        'Random': '#d62728'
    }
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 에피소드별 성능 비교 (큰 그래프)
    ax1 = fig.add_subplot(gs[0, :])
    episodes = range(len(list(results_dict.values())[0]))
    
    for strategy, rewards in results_dict.items():
        ax1.plot(episodes, rewards, label=strategy, 
                color=colors.get(strategy, 'gray'), alpha=0.7, linewidth=2)
    
    ax1.set_title('Episode Performance Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Successful Transmission Slots (NPCA STAs)')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 이동 평균
    ax2 = fig.add_subplot(gs[1, 0])
    window = min(10, len(episodes))
    if window > 1:
        for strategy, rewards in results_dict.items():
            if len(rewards) >= window:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes_ma = range(window-1, len(rewards))
                ax2.plot(episodes_ma, ma, label=strategy, 
                        color=colors.get(strategy, 'gray'), linewidth=2)
    
    ax2.set_title(f'Moving Average (Window={window})', fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 박스플롯
    ax3 = fig.add_subplot(gs[1, 1])
    data = [results_dict[strategy] for strategy in results_dict.keys()]
    labels = list(results_dict.keys())
    
    box_plot = ax3.boxplot(data, labels=labels, patch_artist=True)
    for patch, strategy in zip(box_plot['boxes'], labels):
        patch.set_facecolor(colors.get(strategy, 'gray'))
        patch.set_alpha(0.7)
    
    ax3.set_title('Performance Distribution', fontweight='bold')
    ax3.set_ylabel('Successful Transmission Slots')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 평균 성능 비교
    ax4 = fig.add_subplot(gs[1, 2])
    means = [np.mean(results_dict[strategy]) for strategy in results_dict.keys()]
    stds = [np.std(results_dict[strategy]) for strategy in results_dict.keys()]
    
    bars = ax4.bar(labels, means, yerr=stds, capsize=5,
                   color=[colors.get(strategy, 'gray') for strategy in labels], 
                   alpha=0.7)
    
    ax4.set_title('Average Performance', fontweight='bold')
    ax4.set_ylabel('Mean Successful Transmission Slots')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 바 위에 수치 표시
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    # 5. 채널 점유율 비교
    ax5 = fig.add_subplot(gs[2, 0])
    occupancy_means = details_df.groupby('strategy')['occupancy_ratio'].mean()
    occupancy_stds = details_df.groupby('strategy')['occupancy_ratio'].std()
    
    bars = ax5.bar(occupancy_means.index, occupancy_means.values, 
                   yerr=occupancy_stds.values, capsize=5,
                   color=[colors.get(strategy, 'gray') for strategy in occupancy_means.index], 
                   alpha=0.7)
    
    ax5.set_title('Channel Occupancy Ratio', fontweight='bold')
    ax5.set_ylabel('Occupancy Ratio')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(occupancy_means.values) * 1.2)
    
    # 6. 성능 향상률 히트맵
    ax6 = fig.add_subplot(gs[2, 1:])
    improvement_matrix = []
    strategy_names = list(results_dict.keys())
    
    for base_strategy in strategy_names:
        row = []
        base_mean = np.mean(results_dict[base_strategy])
        for comp_strategy in strategy_names:
            comp_mean = np.mean(results_dict[comp_strategy])
            if base_mean > 0:
                improvement = ((comp_mean - base_mean) / base_mean) * 100
            else:
                improvement = 0
            row.append(improvement)
        improvement_matrix.append(row)
    
    im = ax6.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax6.set_xticks(range(len(strategy_names)))
    ax6.set_yticks(range(len(strategy_names)))
    ax6.set_xticklabels(strategy_names, rotation=45)
    ax6.set_yticklabels(strategy_names)
    ax6.set_title('Performance Improvement Matrix (%)', fontweight='bold')
    ax6.set_xlabel('Compared Strategy')
    ax6.set_ylabel('Base Strategy')
    
    # 히트맵에 수치 표시
    for i in range(len(strategy_names)):
        for j in range(len(strategy_names)):
            ax6.text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                    ha='center', va='center', fontsize=10)
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(results_dict, details_df, save_dir):
    """요약 보고서 생성"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NPCA 성능 비교 테스트 종합 보고서")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 환경 설정
    report_lines.append("테스트 환경:")
    report_lines.append("- NPCA enabled STAs: 10개 (Channel 1, OBSS 발생)")
    report_lines.append("- Non-NPCA STAs: 10개 (Channel 0, OBSS 없음)")
    report_lines.append("- OBSS 생성률: 0.01 (Channel 1)")
    report_lines.append("- OBSS 지속시간: 10-150 slots")
    report_lines.append("- 에피소드 길이: 200 slots")
    report_lines.append("- 테스트 에피소드: 50회")
    report_lines.append("")
    
    # 성능 통계
    report_lines.append("성능 통계 (성공 전송 슬롯 수):")
    report_lines.append("-" * 50)
    
    for strategy, rewards in results_dict.items():
        rewards_array = np.array(rewards)
        report_lines.append(f"{strategy:15s} - Mean: {rewards_array.mean():6.2f}, "
                          f"Std: {rewards_array.std():5.2f}, "
                          f"Max: {rewards_array.max():5.1f}, "
                          f"Min: {rewards_array.min():5.1f}")
    
    report_lines.append("")
    
    # 성능 개선율
    if 'DRL Policy' in results_dict:
        drl_mean = np.mean(results_dict['DRL Policy'])
        report_lines.append("DRL 대비 성능 개선률:")
        report_lines.append("-" * 30)
        
        for strategy, rewards in results_dict.items():
            if strategy != 'DRL Policy':
                strategy_mean = np.mean(rewards)
                if strategy_mean > 0:
                    improvement = ((drl_mean - strategy_mean) / strategy_mean) * 100
                    report_lines.append(f"DRL vs {strategy:15s}: {improvement:+6.1f}%")
        
        report_lines.append("")
    
    # 채널 점유율 분석
    report_lines.append("채널 점유율 분석:")
    report_lines.append("-" * 25)
    
    occupancy_stats = details_df.groupby('strategy')['occupancy_ratio'].agg(['mean', 'std'])
    for strategy in occupancy_stats.index:
        mean_occ = occupancy_stats.loc[strategy, 'mean']
        std_occ = occupancy_stats.loc[strategy, 'std']
        report_lines.append(f"{strategy:15s} - {mean_occ:.3f} ± {std_occ:.3f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 파일 저장
    with open(f"{save_dir}/performance_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return report_lines

def main():
    """메인 함수"""
    print("=" * 80)
    print("NPCA 전략 포괄적 성능 비교 테스트")
    print("환경: 트레이닝과 동일한 설정")
    print("=" * 80)
    
    # 설정
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 50
    num_slots = 200
    results_dir = "./comprehensive_results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"테스트 에피소드 수: {num_episodes}")
    print(f"에피소드당 슬롯 수: {num_slots}")
    print()
    
    # 학습된 모델 로드 시도
    learner = load_trained_model(model_path, device)
    
    # 테스트할 전략들
    strategies = []
    if learner is not None:
        strategies.append(("DRL Policy", learner))
        print("✓ DRL 모델 로드 성공")
    else:
        print("✗ DRL 모델 로드 실패 - DRL 테스트 제외")
    
    strategies.extend([
        ("Always Primary", None),
        ("Always NPCA", None),
        ("Random", None)
    ])
    
    # 전략별 테스트 실행
    results_dict = {}
    all_details = []
    
    for i, (strategy_name, strategy_learner) in enumerate(strategies, 1):
        print(f"\n{i}. {strategy_name} 전략 테스트 중...")
        
        if strategy_name == "DRL Policy":
            rewards, details = run_strategy_test("drl_policy", strategy_learner, num_episodes, num_slots)
        elif strategy_name == "Always Primary":
            rewards, details = run_strategy_test("always_primary", None, num_episodes, num_slots)
        elif strategy_name == "Always NPCA":
            rewards, details = run_strategy_test("always_npca", None, num_episodes, num_slots)
        elif strategy_name == "Random":
            rewards, details = run_strategy_test("random", None, num_episodes, num_slots)
        
        results_dict[strategy_name] = rewards
        all_details.extend(details)
        
        print(f"   평균 성과: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # 결과 저장
    details_df = pd.DataFrame(all_details)
    
    # CSV 저장
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f"{results_dir}/strategy_comparison_results.csv", index=False)
    details_df.to_csv(f"{results_dir}/detailed_results.csv", index=False)
    
    # 시각화
    print("\n결과 시각화 중...")
    plot_comprehensive_results(results_dict, details_df, results_dir)
    
    # 보고서 생성
    print("보고서 생성 중...")
    report_lines = generate_summary_report(results_dict, details_df, results_dir)
    
    # 콘솔 출력
    print("\n" + "\n".join(report_lines))
    
    print(f"\n모든 결과가 {results_dir}/ 에 저장되었습니다.")
    print("- strategy_comparison_results.csv: 전략별 에피소드 결과")
    print("- detailed_results.csv: 상세 분석 데이터") 
    print("- comprehensive_comparison.png: 종합 비교 그래프")
    print("- performance_report.txt: 성능 분석 보고서")

if __name__ == "__main__":
    main()