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
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel
from drl_framework.train import train_semi_mdp

def create_training_config():
    """학습을 위한 설정 생성"""
    
    # 채널 설정
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(20, 40))  # Secondary channel with OBSS
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

def plot_training_results(episode_rewards, episode_losses, save_dir="./results"):
    """학습 결과를 플롯으로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 보상 그래프
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    if episode_losses:
        plt.plot(episode_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {save_dir}/training_results.png")

def main():
    """메인 학습 함수"""
    print("="*60)
    print("Semi-MDP 기반 NPCA STA 학습 시작")
    print("="*60)
    
    # 설정 생성
    channels, stas_config = create_training_config()
    
    # 학습 파라미터 - 높은 STA 밀도로 인한 복잡성 증가로 더 많은 학습 필요
    num_episodes = 3000  # STA 수 증가로 조정
    num_slots_per_episode = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Slots per episode: {num_slots_per_episode}")
    print(f"NPCA enabled STAs: {sum(1 for sta in stas_config if sta['npca_enabled'])}")
    print()
    
    # 학습 실행
    episode_rewards, episode_losses, learner = train_semi_mdp(
        channels=channels,
        stas_config=stas_config, 
        num_episodes=num_episodes,
        num_slots_per_episode=num_slots_per_episode,
        device=device
    )
    
    # 결과 저장 및 출력
    results_dir = "./semi_mdp_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 학습된 모델 저장
    torch.save({
        'policy_net_state_dict': learner.policy_net.state_dict(),
        'target_net_state_dict': learner.target_net.state_dict(),
        'optimizer_state_dict': learner.optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'steps_done': learner.steps_done
    }, f"{results_dir}/semi_mdp_model.pth")
    
    # 그래프 생성
    plot_training_results(episode_rewards, episode_losses, results_dir)
    
    # 최종 통계
    print("\n" + "="*60)
    print("학습 완료!")
    print(f"총 에피소드: {len(episode_rewards)}")
    print(f"평균 보상 (최근 50 에피소드): {sum(episode_rewards[-50:]) / min(50, len(episode_rewards)):.2f}")
    print(f"최대 보상: {max(episode_rewards):.2f}")
    print(f"총 학습 스텝: {learner.steps_done}")
    print(f"메모리 크기: {len(learner.memory)}")
    print(f"모델 저장 위치: {results_dir}/semi_mdp_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()