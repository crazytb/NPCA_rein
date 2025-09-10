#!/usr/bin/env python3
"""
Semi-MDP 기반 NPCA STA 학습을 위한 메인 실행 파일

이 파일은 Semi-MDP를 사용하여 STA의 NPCA 결정을 학습시킵니다.
STA는 Primary 채널이 OBSS로 점유된 상황에서 다음 두 액션 중 선택합니다:
- Action 0: StayPrimary (Primary 채널에서 대기)  
- Action 1: GoNPCA (NPCA 채널로 이동하여 전송)

사용법:
    python main_semi_mdp_training.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel
from drl_framework.train import train_semi_mdp

def create_training_config(obss_duration):
    """학습을 위한 설정 생성"""
    
    # 채널 설정 - OBSS duration을 고정값으로 설정
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.01, obss_duration_range=(obss_duration, obss_duration))  # 고정된 OBSS duration
    ]
    
    # STA 설정 - 각 채널에 10개씩 STA 배치
    stas_config = []
    
    # Channel 1의 STA들 (NPCA 지원) - 10개
    for i in range(10):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    # Channel 0의 STA들 (기존 방식) - 10개
    for i in range(10, 20):
        stas_config.append({
            "sta_id": i,
            "channel_id": 0, 
            "npca_enabled": False,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    return channels, stas_config

def calculate_running_average(data, window_size):
    """주어진 데이터에 대한 이동 평균을 계산합니다."""
    if len(data) < window_size:
        return data
    
    running_avg = []
    for i in range(len(data)):
        if i < window_size - 1:
            # 초기 구간에서는 사용 가능한 모든 데이터 평균 사용
            running_avg.append(np.mean(data[:i+1]))
        else:
            # 윈도우 크기만큼의 구간 평균 사용
            running_avg.append(np.mean(data[i-window_size+1:i+1]))
    
    return running_avg

def plot_training_results(episode_rewards, episode_losses, save_dir="./results", reward_window=100, loss_window=50, obss_duration=None):
    """학습 결과를 플롯으로 저장 (러닝 평균 오버레이 포함)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 보상 러닝 평균 계산
    reward_running_avg = calculate_running_average(episode_rewards, reward_window)
    
    # 보상 그래프
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    # 원본 데이터 (반투명)
    plt.plot(episode_rewards, alpha=0.3, color='lightblue', label='Raw Rewards')
    # 러닝 평균 (진한 색상)
    plt.plot(reward_running_avg, color='darkblue', linewidth=2, label=f'Running Avg (window={reward_window})')
    title = 'Episode Rewards with Running Average'
    if obss_duration:
        title += f' (OBSS Duration: {obss_duration})'
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    if episode_losses:
        # 손실 러닝 평균 계산
        loss_running_avg = calculate_running_average(episode_losses, loss_window)
        
        # 원본 데이터 (반투명)
        plt.plot(episode_losses, alpha=0.3, color='lightcoral', label='Raw Loss')
        # 러닝 평균 (진한 색상)
        plt.plot(loss_running_avg, color='darkred', linewidth=2, label=f'Running Avg (window={loss_window})')
        title = 'Training Loss with Running Average'
        if obss_duration:
            title += f' (OBSS Duration: {obss_duration})'
        plt.title(title)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {save_dir}/training_results.png")
    print(f"Running averages: Rewards (window={reward_window}), Loss (window={loss_window})")

def run_experiment(obss_duration, experiment_name):
    """단일 OBSS duration 값에 대한 실험 실행"""
    print(f"실험 시작: {experiment_name} (OBSS Duration: {obss_duration})")
    print("-" * 50)
    
    # 설정 생성
    channels, stas_config = create_training_config(obss_duration)
    
    # 학습 파라미터
    num_episodes = 5000  # 비교를 위해 에피소드 수를 줄임
    num_slots_per_episode = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 학습 실행
    episode_rewards, episode_losses, learner = train_semi_mdp(
        channels=channels,
        stas_config=stas_config, 
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots_per_episode,
        device=device
    )
    
    # 결과 저장
    results_dir = f"./obss_comparison_results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 학습된 모델 저장
    torch.save({
        'policy_net_state_dict': learner.policy_net.state_dict(),
        'target_net_state_dict': learner.target_net.state_dict(),
        'optimizer_state_dict': learner.optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'steps_done': learner.steps_done,
        'obss_duration': obss_duration
    }, f"{results_dir}/model.pth")
    
    # 그래프 생성
    plot_training_results(episode_rewards, episode_losses, results_dir, obss_duration=obss_duration)
    
    # 실험 통계
    final_avg_reward = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
    max_reward = max(episode_rewards)
    
    print(f"실험 완료: {experiment_name}")
    print(f"평균 보상 (최근 50 에피소드): {final_avg_reward:.2f}")
    print(f"최대 보상: {max_reward:.2f}")
    print(f"총 학습 스텝: {learner.steps_done}")
    print("-" * 50)
    print()
    
    return {
        'experiment_name': experiment_name,
        'obss_duration': obss_duration,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_avg_reward': final_avg_reward,
        'max_reward': max_reward,
        'steps_done': learner.steps_done
    }

def plot_comparison_results(experiment_results):
    """모든 실험 결과를 비교하는 그래프 생성"""
    plt.figure(figsize=(20, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. 에피소드별 보상 비교 (러닝 평균)
    plt.subplot(2, 3, 1)
    for i, result in enumerate(experiment_results):
        rewards = result['episode_rewards']
        running_avg = calculate_running_average(rewards, 50)
        plt.plot(running_avg, color=colors[i], linewidth=2, 
                label=f"OBSS Duration: {result['obss_duration']}")
    plt.title('Episode Rewards Comparison (Running Average)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 최종 성능 비교 (최근 50 에피소드 평균)
    plt.subplot(2, 3, 2)
    durations = [result['obss_duration'] for result in experiment_results]
    final_rewards = [result['final_avg_reward'] for result in experiment_results]
    plt.bar(range(len(durations)), final_rewards, color=colors[:len(durations)])
    plt.title('Final Performance Comparison')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Average Reward (Last 50 Episodes)')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(final_rewards):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 3. 최대 보상 비교
    plt.subplot(2, 3, 3)
    max_rewards = [result['max_reward'] for result in experiment_results]
    plt.bar(range(len(durations)), max_rewards, color=colors[:len(durations)])
    plt.title('Maximum Reward Comparison')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Maximum Reward')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(max_rewards):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 4. 학습 진행 곡선 비교 (처음 200 에피소드)
    plt.subplot(2, 3, 4)
    for i, result in enumerate(experiment_results):
        rewards = result['episode_rewards'][:200]
        running_avg = calculate_running_average(rewards, 20)
        plt.plot(running_avg, color=colors[i], linewidth=2, 
                label=f"OBSS Duration: {result['obss_duration']}")
    plt.title('Early Learning Progress (First 200 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 수렴 속도 비교 (보상이 특정 임계값을 넘는 에피소드)
    plt.subplot(2, 3, 5)
    convergence_episodes = []
    threshold = 10.0  # 임계 보상값
    
    for result in experiment_results:
        rewards = result['episode_rewards']
        running_avg = calculate_running_average(rewards, 20)
        convergence_ep = next((i for i, r in enumerate(running_avg) if r >= threshold), len(rewards))
        convergence_episodes.append(convergence_ep)
    
    plt.bar(range(len(durations)), convergence_episodes, color=colors[:len(durations)])
    plt.title(f'Convergence Speed (Episodes to reach {threshold} reward)')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Episodes to Convergence')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(convergence_episodes):
        plt.text(i, v + 5, f'{v}' if v < len(experiment_results[i]['episode_rewards']) else 'N/A', ha='center')
    plt.grid(True, alpha=0.3)
    
    # 6. 학습 안정성 비교 (최근 100 에피소드 표준편차)
    plt.subplot(2, 3, 6)
    stability = []
    for result in experiment_results:
        recent_rewards = result['episode_rewards'][-100:]
        stability.append(np.std(recent_rewards))
    
    plt.bar(range(len(durations)), stability, color=colors[:len(durations)])
    plt.title('Learning Stability (Std of Last 100 Episodes)')
    plt.xlabel('OBSS Duration')
    plt.ylabel('Standard Deviation')
    plt.xticks(range(len(durations)), durations)
    for i, v in enumerate(stability):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    comparison_dir = "./obss_comparison_results"
    plt.savefig(f"{comparison_dir}/obss_duration_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"비교 결과 저장됨: {comparison_dir}/obss_duration_comparison.png")

def main():
    """메인 학습 함수 - OBSS duration 비교 실험"""
    print("="*60)
    print("OBSS Duration 비교 실험 시작")
    print("="*60)
    
    # 테스트할 OBSS duration 값들 (slots 단위)
    obss_durations = [20, 50, 100, 150]  # 짧음, 중간-짧음, 중간-긴, 긴
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"테스트할 OBSS duration 값들: {obss_durations}")
    print()
    
    # 모든 실험 결과 저장
    all_results = []
    
    # 각 OBSS duration에 대해 실험 실행
    for obss_duration in obss_durations:
        experiment_name = f"obss_{obss_duration}_slots"
        result = run_experiment(obss_duration, experiment_name)
        all_results.append(result)
    
    # 비교 결과 시각화
    plot_comparison_results(all_results)
    
    # 종합 분석 출력
    print("\n" + "="*60)
    print("OBSS Duration 비교 실험 완료!")
    print("="*60)
    print("실험 결과 요약:")
    print("-" * 40)
    
    for result in all_results:
        print(f"OBSS Duration {result['obss_duration']:3d} slots: "
              f"최종 평균 보상 = {result['final_avg_reward']:6.2f}, "
              f"최대 보상 = {result['max_reward']:6.2f}")
    
    # 최적 OBSS duration 식별
    best_result = max(all_results, key=lambda x: x['final_avg_reward'])
    print(f"\n최적 OBSS Duration: {best_result['obss_duration']} slots "
          f"(평균 보상: {best_result['final_avg_reward']:.2f})")
    
    print("="*60)

if __name__ == "__main__":
    main()