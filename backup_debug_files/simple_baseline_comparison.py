#!/usr/bin/env python3
"""
간단한 베이스라인 비교: DRL vs Always NPCA vs Always Primary

기존 test_model.py를 활용하여 세 가지 전략 비교:
1. drl: 학습된 DRL 정책  
2. offload_only: Always NPCA (GoNPCA만 선택)
3. local_only: Always Primary (StayPrimary만 선택)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_model import test_model

def run_comparison(num_episodes=100):
    """세 가지 전략으로 성능 비교 실행"""
    results = {}
    
    strategies = {
        'DRL Policy': 'drl',
        'Always NPCA': 'offload_only', 
        'Always Primary': 'local_only'
    }
    
    print("="*60)
    print("성능 비교 테스트 시작")
    print("="*60)
    
    for strategy_name, mode in strategies.items():
        print(f"\n{strategy_name} 테스트 중...")
        
        # test_model 함수 호출하여 결과 가져오기
        try:
            episode_rewards, decision_log = test_model(mode=mode, num_episodes=num_episodes)
            
            results[strategy_name] = {
                'rewards': episode_rewards,
                'decisions': decision_log,
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards),
                'max': np.max(episode_rewards),
                'min': np.min(episode_rewards)
            }
            
            print(f"{strategy_name} 완료 - 평균 보상: {results[strategy_name]['mean']:.3f}")
            
        except Exception as e:
            print(f"{strategy_name} 테스트 실패: {e}")
            continue
    
    return results

def plot_comparison_results(results, save_dir="./comparison_results"):
    """비교 결과 플롯 생성"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 색상 설정
    colors = {'DRL Policy': 'blue', 'Always NPCA': 'red', 'Always Primary': 'green'}
    
    # 1. 에피소드별 보상
    ax1 = axes[0, 0]
    for strategy_name, data in results.items():
        episodes = range(len(data['rewards']))
        ax1.plot(episodes, data['rewards'], label=strategy_name, 
                alpha=0.7, color=colors.get(strategy_name, 'black'))
    ax1.set_title('Episode Rewards Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Running Average (window=20)
    ax2 = axes[0, 1]
    window = 20
    for strategy_name, data in results.items():
        rewards = data['rewards']
        running_avg = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
        episodes = range(len(running_avg))
        ax2.plot(episodes, running_avg, label=f'{strategy_name}', 
                linewidth=2, color=colors.get(strategy_name, 'black'))
    ax2.set_title(f'Running Average (window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 보상 분포 박스플롯
    ax3 = axes[1, 0]
    data_list = [results[name]['rewards'] for name in results.keys()]
    labels = list(results.keys())
    bp = ax3.boxplot(data_list, labels=labels, patch_artist=True)
    
    # 박스플롯 색상 설정
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, 'lightblue'))
        patch.set_alpha(0.7)
    
    ax3.set_title('Reward Distribution')
    ax3.set_ylabel('Total Reward')
    ax3.grid(True, alpha=0.3)
    
    # 4. 성능 비교 막대그래프
    ax4 = axes[1, 1]
    strategy_names = list(results.keys())
    means = [results[name]['mean'] for name in strategy_names]
    stds = [results[name]['std'] for name in strategy_names]
    
    bars = ax4.bar(strategy_names, means, yerr=stds, capsize=5, 
                   color=[colors.get(name, 'lightblue') for name in strategy_names],
                   alpha=0.7, edgecolor='black')
    ax4.set_title('Average Performance Comparison')
    ax4.set_ylabel('Mean Reward')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 값 표시
    for bar, mean in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to {save_dir}/strategy_comparison.png")

def save_results_csv(results, save_dir="./comparison_results"):
    """결과를 CSV 파일로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 보상 데이터를 DataFrame으로 변환
    max_length = max(len(data['rewards']) for data in results.values())
    
    df_data = {}
    for strategy_name, data in results.items():
        rewards = data['rewards']
        # 길이를 맞추기 위해 패딩
        padded_rewards = rewards + [np.nan] * (max_length - len(rewards))
        df_data[strategy_name] = padded_rewards
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{save_dir}/rewards_comparison.csv", index=False)
    
    # 통계 요약을 별도 CSV로 저장
    summary_data = []
    for strategy_name, data in results.items():
        summary_data.append({
            'Strategy': strategy_name,
            'Mean': data['mean'],
            'Std': data['std'],
            'Max': data['max'],
            'Min': data['min'],
            'Episodes': len(data['rewards'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{save_dir}/performance_summary.csv", index=False)
    
    print(f"Results saved to {save_dir}/rewards_comparison.csv")
    print(f"Summary saved to {save_dir}/performance_summary.csv")

def print_detailed_comparison(results):
    """상세 비교 결과 출력"""
    print("\n" + "="*80)
    print("상세 성능 비교 결과")
    print("="*80)
    
    # 성능 통계 출력
    for strategy_name, data in results.items():
        print(f"\n{strategy_name}:")
        print(f"  평균 보상: {data['mean']:.4f}")
        print(f"  표준편차:   {data['std']:.4f}")  
        print(f"  최대 보상: {data['max']:.4f}")
        print(f"  최소 보상: {data['min']:.4f}")
        print(f"  에피소드 수: {len(data['rewards'])}")
    
    # 상대적 성능 비교
    if 'DRL Policy' in results and 'Always NPCA' in results:
        drl_mean = results['DRL Policy']['mean']
        npca_mean = results['Always NPCA']['mean']
        improvement = ((drl_mean - npca_mean) / npca_mean) * 100
        print(f"\nDRL vs Always NPCA 성능 개선: {improvement:+.1f}%")
    
    if 'DRL Policy' in results and 'Always Primary' in results:
        drl_mean = results['DRL Policy']['mean']
        primary_mean = results['Always Primary']['mean']
        improvement = ((drl_mean - primary_mean) / primary_mean) * 100
        print(f"DRL vs Always Primary 성능 개선: {improvement:+.1f}%")
    
    # 베스트 전략 식별
    best_strategy = max(results.keys(), key=lambda k: results[k]['mean'])
    best_mean = results[best_strategy]['mean']
    print(f"\n최고 성능 전략: {best_strategy} (평균 보상: {best_mean:.4f})")
    
    print("="*80)

def main():
    """메인 비교 함수"""
    # 비교 실행 (테스트용으로 50 에피소드)
    results = run_comparison(num_episodes=50)
    
    if not results:
        print("비교 테스트 실패 - 결과가 없습니다.")
        return
    
    # 결과 저장
    save_results_csv(results)
    
    # 플롯 생성
    plot_comparison_results(results)
    
    # 상세 결과 출력  
    print_detailed_comparison(results)

if __name__ == "__main__":
    main()