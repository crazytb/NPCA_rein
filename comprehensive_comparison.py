#!/usr/bin/env python3
"""
제안기법(DRL) vs Always NPCA vs Always Primary 종합 비교
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.train import SemiMDPLearner

def test_strategy(strategy_name, channels, stas_config, num_episodes=50, device="cpu", model_path=None):
    """단일 전략 성능 테스트"""
    print(f"\n{strategy_name} 테스트 중...")
    
    episode_rewards = []
    decision_log = []
    occupancy_times = []
    
    # DRL 정책인 경우 모델 로드
    learner = None
    if strategy_name == "DRL Policy" and model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        learner = SemiMDPLearner(n_observations=4, n_actions=2, device=device)
        learner.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        learner.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learner.steps_done = checkpoint['steps_done']
        learner.policy_net.eval()
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"  Episode {episode}/{num_episodes}")
        
        # 새로운 채널 인스턴스 (각 에피소드마다 독립적인 OBSS 패턴)
        episode_channels = [
            Channel(channel_id=0, obss_generation_rate=channels[0].obss_generation_rate),
            Channel(channel_id=1, obss_generation_rate=channels[1].obss_generation_rate, 
                   obss_duration_range=channels[1].obss_duration_range)
        ]
        
        # STA 생성
        stas = []
        for config in stas_config:
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=episode_channels[config["channel_id"]],
                npca_channel=episode_channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=config.get("npca_enabled", False),
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", 33),
                learner=learner
            )
            
            # 고정 전략 설정
            if strategy_name == "Always NPCA":
                sta._fixed_action = 1
            elif strategy_name == "Always Primary":
                sta._fixed_action = 0
            
            sta.decision_log = decision_log
            sta.current_episode = episode
            stas.append(sta)
        
        # 시뮬레이션 실행
        simulator = Simulator(num_slots=200, channels=episode_channels, stas=stas)
        if learner:
            simulator.memory = learner.memory
            simulator.device = device
        simulator.run()
        
        # 결과 수집
        total_reward = sum(sta.episode_reward for sta in stas)
        total_occupancy = sum(sta.channel_occupancy_time for sta in stas)
        
        episode_rewards.append(total_reward)
        occupancy_times.append(total_occupancy)
    
    return {
        'rewards': episode_rewards,
        'occupancy_times': occupancy_times,
        'decisions': decision_log,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_occupancy': np.mean(occupancy_times),
        'success_rate': (np.array(episode_rewards) > 0).mean() * 100
    }

def run_comprehensive_comparison():
    """종합 성능 비교 실행"""
    print("="*80)
    print("제안기법(DRL) vs Always NPCA vs Always Primary 종합 비교")
    print("="*80)
    
    # 테스트 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 100
    
    # 채널 설정 (높은 간섭 환경)
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # NPCA channel
        Channel(channel_id=1, obss_generation_rate=0.6, obss_duration_range=(80, 150))  # Primary channel
    ]
    
    # STA 설정 (10개 STA)
    stas_config = []
    for i in range(10):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    print(f"테스트 설정:")
    print(f"  Device: {device}")
    print(f"  Episodes: {num_episodes}")
    print(f"  STAs: {len(stas_config)}")
    print(f"  OBSS rate: {channels[1].obss_generation_rate}")
    print(f"  OBSS duration: {channels[1].obss_duration_range}")
    print(f"  Episode length: 200 slots")
    
    # 세 가지 전략 테스트
    strategies = [
        ("DRL Policy (제안기법)", channels, model_path),
        ("Always NPCA", channels, None),
        ("Always Primary", channels, None)
    ]
    
    results = {}
    
    for strategy_name, test_channels, model in strategies:
        result = test_strategy(
            strategy_name, test_channels, stas_config, 
            num_episodes, device, model
        )
        results[strategy_name] = result
    
    return results, channels, stas_config

def save_results_and_plots(results, save_dir="./comprehensive_results"):
    """결과 저장 및 플롯 생성"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. CSV 파일 저장
    df_data = {}
    for name, data in results.items():
        df_data[f"{name}_reward"] = data['rewards']
        df_data[f"{name}_occupancy"] = data['occupancy_times']
    
    # 길이를 맞추기 위해 패딩
    max_length = max(len(data['rewards']) for data in results.values())
    for key in df_data:
        if len(df_data[key]) < max_length:
            df_data[key].extend([np.nan] * (max_length - len(df_data[key])))
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{save_dir}/comparison_results.csv", index=False)
    
    # 2. 통계 요약 저장
    summary_data = []
    for name, data in results.items():
        summary_data.append({
            'Strategy': name,
            'Mean_Reward_%': data['mean_reward'],
            'Std_Reward_%': data['std_reward'],
            'Mean_Occupancy_slots': data['mean_occupancy'],
            'Success_Rate_%': data['success_rate'],
            'Episodes': len(data['rewards'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{save_dir}/performance_summary.csv", index=False)
    
    # 3. 상세 플롯 생성
    create_detailed_plots(results, save_dir)
    
    print(f"결과 저장 완료: {save_dir}/")

def create_detailed_plots(results, save_dir):
    """상세 플롯 생성"""
    plt.style.use('default')
    
    # 색상 설정
    colors = {
        'DRL Policy (제안기법)': '#2E86AB',
        'Always NPCA': '#A23B72', 
        'Always Primary': '#F18F01'
    }
    
    # 큰 플롯 생성 (3x2 레이아웃)
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 에피소드별 점유율
    plt.subplot(3, 2, 1)
    for name, data in results.items():
        episodes = range(len(data['rewards']))
        plt.plot(episodes, data['rewards'], label=name, alpha=0.7, 
                color=colors.get(name, 'gray'), linewidth=1)
    plt.title('Episode Occupancy Rate (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Occupancy Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Running Average (window=20)
    plt.subplot(3, 2, 2)
    window = 20
    for name, data in results.items():
        rewards = data['rewards']
        running_avg = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
        plt.plot(running_avg, label=name, linewidth=2.5, color=colors.get(name, 'gray'))
    plt.title(f'Running Average (window={window})', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Average Occupancy Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 박스플롯
    plt.subplot(3, 2, 3)
    data_list = [results[name]['rewards'] for name in results.keys()]
    labels = [name.replace(' (제안기법)', '\n(제안기법)') for name in results.keys()]
    bp = plt.boxplot(data_list, labels=labels, patch_artist=True)
    
    for i, (patch, label) in enumerate(zip(bp['boxes'], results.keys())):
        patch.set_facecolor(colors.get(label, 'lightblue'))
        patch.set_alpha(0.7)
    
    plt.title('Occupancy Rate Distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Occupancy Rate (%)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. 성능 비교 막대그래프
    plt.subplot(3, 2, 4)
    names = [name.replace(' (제안기법)', '\n(제안기법)') for name in results.keys()]
    means = [results[name]['mean_reward'] for name in results.keys()]
    stds = [results[name]['std_reward'] for name in results.keys()]
    
    bars = plt.bar(names, means, yerr=stds, capsize=8, 
                  color=[colors.get(name, 'lightblue') for name in results.keys()],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Average Performance Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Occupancy Rate (%)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 5. 성공률 비교
    plt.subplot(3, 2, 5)
    success_rates = [results[name]['success_rate'] for name in results.keys()]
    bars = plt.bar(names, success_rates, 
                  color=[colors.get(name, 'lightblue') for name in results.keys()],
                  alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Success Rate Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('Episodes with Occupancy > 0 (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. 총 점유 시간 비교
    plt.subplot(3, 2, 6)
    occupancy_means = [results[name]['mean_occupancy'] for name in results.keys()]
    bars = plt.bar(names, occupancy_means,
                  color=[colors.get(name, 'lightblue') for name in results.keys()],
                  alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Average Total Occupancy Time', fontsize=12, fontweight='bold')
    plt.ylabel('Total Occupancy (slots)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, occ in zip(bars, occupancy_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{occ:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(results):
    """상세 결과 출력"""
    print("\n" + "="*80)
    print("상세 성능 비교 결과")
    print("="*80)
    
    for name, data in results.items():
        print(f"\n📊 {name}:")
        print(f"   평균 점유율: {data['mean_reward']:.2f}% (±{data['std_reward']:.2f})")
        print(f"   평균 점유 시간: {data['mean_occupancy']:.1f} slots")
        print(f"   성공률: {data['success_rate']:.1f}%")
        print(f"   최대 점유율: {max(data['rewards']):.1f}%")
        print(f"   최소 점유율: {min(data['rewards']):.1f}%")
    
    # 상대 성능 비교
    print(f"\n" + "="*50)
    print("상대 성능 비교")
    print("="*50)
    
    drl_mean = results['DRL Policy (제안기법)']['mean_reward']
    npca_mean = results['Always NPCA']['mean_reward']
    primary_mean = results['Always Primary']['mean_reward']
    
    if npca_mean > 0:
        drl_vs_npca = ((drl_mean - npca_mean) / npca_mean) * 100
        print(f"제안기법 vs Always NPCA: {drl_vs_npca:+.1f}%")
    
    if primary_mean > 0:
        drl_vs_primary = ((drl_mean - primary_mean) / primary_mean) * 100
        print(f"제안기법 vs Always Primary: {drl_vs_primary:+.1f}%")
    else:
        print(f"제안기법 vs Always Primary: +∞% (Always Primary = 0%)")
    
    if npca_mean > 0 and primary_mean > 0:
        npca_vs_primary = ((npca_mean - primary_mean) / primary_mean) * 100
        print(f"Always NPCA vs Always Primary: {npca_vs_primary:+.1f}%")
    
    # 최고 성능 전략
    best_strategy = max(results.keys(), key=lambda k: results[k]['mean_reward'])
    best_mean = results[best_strategy]['mean_reward']
    print(f"\n🏆 최고 성능: {best_strategy} ({best_mean:.2f}% 점유율)")

def main():
    """메인 실행 함수"""
    # 종합 비교 실행
    results, channels, stas_config = run_comprehensive_comparison()
    
    # 결과 저장 및 플롯
    save_results_and_plots(results)
    
    # 상세 결과 출력
    print_detailed_results(results)
    
    print("\n" + "="*80)
    print("비교 완료! 결과는 ./comprehensive_results/ 폴더에 저장되었습니다.")
    print("="*80)

if __name__ == "__main__":
    main()