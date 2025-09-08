#!/usr/bin/env python3
"""
공정한 성능 비교: 모든 에피소드에서 동일한 OBSS 패턴 보장
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.train import SemiMDPLearner

def create_test_config_with_guaranteed_obss():
    """OBSS 보장된 테스트 설정"""
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # NPCA channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.8, obss_duration_range=(80, 150))  # Primary channel with high OBSS
    ]
    
    stas_config = []
    for i in range(10):
        stas_config.append({
            "sta_id": i,
            "channel_id": 1,
            "npca_enabled": True,
            "ppdu_duration": 33,
            "radio_transition_time": 1
        })
    
    return channels, stas_config

def test_strategy_with_guaranteed_obss(channels, stas_config, strategy="drl", model_path=None, num_episodes=50, device="cpu"):
    """OBSS 보장된 환경에서 전략 테스트"""
    
    episode_rewards = []
    decision_log = []
    
    # DRL 정책인 경우 모델 로드
    learner = None
    if strategy == "drl" and model_path and os.path.exists(model_path):
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
        
        # 각 에피소드마다 새 STA 생성
        stas = []
        for config in stas_config:
            sta = STA(
                sta_id=config["sta_id"],
                channel_id=config["channel_id"],
                primary_channel=channels[config["channel_id"]],
                npca_channel=channels[0] if config["channel_id"] == 1 else None,
                npca_enabled=config.get("npca_enabled", False),
                radio_transition_time=config.get("radio_transition_time", 1),
                ppdu_duration=config.get("ppdu_duration", 33),
                learner=learner
            )
            
            # 고정 전략 설정
            if strategy == "always_npca":
                sta._fixed_action = 1
            elif strategy == "always_primary":
                sta._fixed_action = 0
            
            sta.decision_log = decision_log
            sta.current_episode = episode
            stas.append(sta)
        
        # 시뮬레이터 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=stas)
        if learner:
            simulator.memory = learner.memory
            simulator.device = device
        simulator.run()
        
        # 보상 수집
        total_reward = sum(sta.episode_reward for sta in stas)
        episode_rewards.append(total_reward / 100)  # 정규화
    
    return episode_rewards, decision_log

def run_fair_comparison(num_episodes=50):
    """공정한 비교 실행"""
    print("="*60)
    print("공정한 성능 비교 (OBSS 보장)")
    print("="*60)
    
    channels, stas_config = create_test_config_with_guaranteed_obss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    
    print(f"Device: {device}")
    print(f"Test episodes: {num_episodes}")
    print(f"OBSS rate increased to: {channels[1].obss_generation_rate}")
    print(f"OBSS duration range: {channels[1].obss_duration_range}")
    print()
    
    results = {}
    
    # 1. DRL Policy
    print("DRL Policy 테스트 중...")
    drl_rewards, drl_log = test_strategy_with_guaranteed_obss(
        channels, stas_config, "drl", model_path, num_episodes, device)
    results['DRL Policy'] = drl_rewards
    
    # 2. Always NPCA
    print("Always NPCA 테스트 중...")
    npca_rewards, npca_log = test_strategy_with_guaranteed_obss(
        channels, stas_config, "always_npca", None, num_episodes, device)
    results['Always NPCA'] = npca_rewards
    
    # 3. Always Primary
    print("Always Primary 테스트 중...")
    primary_rewards, primary_log = test_strategy_with_guaranteed_obss(
        channels, stas_config, "always_primary", None, num_episodes, device)
    results['Always Primary'] = primary_rewards
    
    # 결과 저장
    results_dir = "./fair_comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(f"{results_dir}/fair_rewards_comparison.csv", index=False)
    
    # 결과 출력
    print("\n" + "="*60)
    print("공정한 성능 비교 결과")
    print("="*60)
    
    for name, rewards in results.items():
        rewards_array = np.array(rewards)
        non_zero_count = (rewards_array > 0).sum()
        print(f"{name:15} - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}, Max: {np.max(rewards):.3f}")
        print(f"                  Non-zero episodes: {non_zero_count}/{len(rewards)} ({non_zero_count/len(rewards)*100:.1f}%)")
    
    # 상대 성능 계산
    if np.mean(results['Always NPCA']) > 0:
        drl_vs_npca = ((np.mean(results['DRL Policy']) - np.mean(results['Always NPCA'])) / np.mean(results['Always NPCA'])) * 100
        print(f"\nDRL vs Always NPCA 개선: {drl_vs_npca:+.1f}%")
    
    if np.mean(results['Always Primary']) > 0:
        drl_vs_primary = ((np.mean(results['DRL Policy']) - np.mean(results['Always Primary'])) / np.mean(results['Always Primary'])) * 100
        print(f"DRL vs Always Primary 개선: {drl_vs_primary:+.1f}%")
    
    print("="*60)
    
    # 간단한 플롯
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for name, rewards in results.items():
        plt.plot(rewards, label=name, alpha=0.7)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    data_list = [results[name] for name in results.keys()]
    labels = list(results.keys())
    plt.boxplot(data_list, labels=labels)
    plt.title('Reward Distribution')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    means = [np.mean(results[name]) for name in results.keys()]
    stds = [np.std(results[name]) for name in results.keys()]
    plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title('Average Performance')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 4)
    success_rates = [(np.array(results[name]) > 0).mean() * 100 for name in results.keys()]
    plt.bar(labels, success_rates, alpha=0.7)
    plt.title('Success Rate (%)')
    plt.ylabel('Episodes with Reward > 0 (%)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/fair_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {results_dir}/")

if __name__ == "__main__":
    run_fair_comparison(50)