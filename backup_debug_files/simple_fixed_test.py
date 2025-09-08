#!/usr/bin/env python3
"""
간단한 고정 전략 테스트
"""

import torch
import numpy as np
from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.train import SemiMDPLearner

def test_single_strategy(strategy_name, fixed_action, num_episodes=10):
    """단일 전략 테스트"""
    print(f"\n{strategy_name} 테스트 ({num_episodes} 에피소드):")
    
    # 높은 OBSS 발생률로 설정
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # NPCA channel
        Channel(channel_id=1, obss_generation_rate=0.6, obss_duration_range=(80, 150))  # Primary channel
    ]
    
    episode_rewards = []
    decision_counts = []
    
    for episode in range(num_episodes):
        # STA 생성
        sta = STA(
            sta_id=0,
            channel_id=1,
            primary_channel=channels[1],
            npca_channel=channels[0],
            npca_enabled=True,
            ppdu_duration=33,
            radio_transition_time=1,
            learner=None
        )
        
        # 고정 전략 설정
        if fixed_action is not None:
            sta._fixed_action = fixed_action
        
        decision_log = []
        sta.decision_log = decision_log
        sta.current_episode = episode
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
        simulator.run()
        
        # 결과 수집
        episode_rewards.append(sta.episode_reward)
        decision_counts.append(len(decision_log))
        
        if episode < 3:  # 처음 3개 에피소드 상세 출력
            print(f"  Episode {episode}: Reward={sta.episode_reward:.1f}, Decisions={len(decision_log)}")
            if decision_log:
                for i, log in enumerate(decision_log):
                    print(f"    Decision {i+1}: Slot={log['slot']}, Action={log['action']}, OBSS_remain={log['primary_channel_obss_occupied_remained']}")
    
    # 통계 출력
    rewards = np.array(episode_rewards)
    decisions = np.array(decision_counts)
    print(f"  평균 보상: {rewards.mean():.2f} (std: {rewards.std():.2f})")
    print(f"  평균 결정 횟수: {decisions.mean():.1f}")
    print(f"  성공 에피소드: {(rewards > 0).sum()}/{num_episodes}")
    
    return episode_rewards

def test_drl_policy(num_episodes=10):
    """DRL 정책 테스트"""
    print(f"\nDRL Policy 테스트 ({num_episodes} 에피소드):")
    
    # 모델 로드
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    learner = SemiMDPLearner(n_observations=4, n_actions=2, device='cpu')
    learner.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    learner.steps_done = checkpoint['steps_done']
    learner.policy_net.eval()
    
    # 채널 설정 (고정 전략과 동일)
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),
        Channel(channel_id=1, obss_generation_rate=0.6, obss_duration_range=(80, 150))
    ]
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # STA 생성 (learner 포함)
        sta = STA(
            sta_id=0,
            channel_id=1,
            primary_channel=channels[1],
            npca_channel=channels[0],
            npca_enabled=True,
            ppdu_duration=33,
            radio_transition_time=1,
            learner=learner
        )
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=[sta])
        simulator.memory = learner.memory
        simulator.device = 'cpu'
        simulator.run()
        
        episode_rewards.append(sta.episode_reward)
        
        if episode < 3:
            print(f"  Episode {episode}: Reward={sta.episode_reward:.1f}")
    
    # 통계 출력
    rewards = np.array(episode_rewards)
    print(f"  평균 보상: {rewards.mean():.2f} (std: {rewards.std():.2f})")
    print(f"  성공 에피소드: {(rewards > 0).sum()}/{num_episodes}")
    
    return episode_rewards

if __name__ == "__main__":
    print("="*50)
    print("간단한 전략 비교 테스트")
    print("="*50)
    
    # 고정 전략들 테스트
    always_npca_rewards = test_single_strategy("Always NPCA", 1, 10)
    always_primary_rewards = test_single_strategy("Always Primary", 0, 10)
    
    # DRL 정책 테스트
    drl_rewards = test_drl_policy(10)
    
    # 비교 결과
    print("\n" + "="*50)
    print("비교 결과:")
    print(f"DRL Policy:     평균 {np.mean(drl_rewards):.2f}")
    print(f"Always NPCA:    평균 {np.mean(always_npca_rewards):.2f}")
    print(f"Always Primary: 평균 {np.mean(always_primary_rewards):.2f}")
    print("="*50)