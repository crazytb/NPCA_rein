#!/usr/bin/env python3
"""
DRL vs Always NPCA 직접 성능 비교

Semi-MDP 환경에서 두 전략의 성능을 직접 비교합니다:
1. 학습된 DRL 정책 
2. Always NPCA (항상 GoNPCA 선택)
3. Always Primary (항상 StayPrimary 선택)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drl_framework.random_access import Channel, Simulator
from drl_framework.network import DQN

def create_test_config():
    """테스트를 위한 설정 생성 (훈련과 동일)"""
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.05, obss_duration_range=(80, 150))  # NPCA channel  
    ]
    
    # Channel 1의 NPCA 지원 STA들만 테스트 (10개)
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

def test_drl_policy(channels, stas_config, model_path, num_episodes=100, device="cpu"):
    """학습된 DRL 정책 테스트"""
    print("DRL Policy 테스트 중...")
    
    # 모델 로드
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일이 없습니다 - {model_path}")
        return [], []
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 학습된 모델 생성
    from drl_framework.train import SemiMDPLearner
    learner = SemiMDPLearner(n_observations=4, n_actions=2, device=device)
    learner.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    learner.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    learner.steps_done = checkpoint['steps_done']
    learner.policy_net.eval()
    
    episode_rewards = []
    decision_log = []
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"  Episode {episode}/{num_episodes}")
        
        # 각 에피소드마다 STA들을 새로 생성
        from drl_framework.random_access import STA
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
            sta.decision_log = decision_log
            sta.current_episode = episode
            stas.append(sta)
        
        # 시뮬레이터 생성 및 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=stas)
        simulator.memory = learner.memory
        simulator.device = device
        simulator.run()
        
        # 에피소드 보상 수집
        total_reward = sum(sta.episode_reward for sta in stas)
        episode_rewards.append(total_reward / 100)  # 정규화
    
    return episode_rewards, decision_log

def test_fixed_strategy(channels, stas_config, strategy="always_npca", num_episodes=100, device="cpu"):
    """고정 전략 테스트"""
    strategy_name = "Always NPCA" if strategy == "always_npca" else "Always Primary" 
    print(f"{strategy_name} 테스트 중...")
    
    episode_rewards = []
    decision_log = []
    
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"  Episode {episode}/{num_episodes}")
        
        # 각 에피소드마다 STA들을 새로 생성 (학습자 없이)
        from drl_framework.random_access import STA
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
                learner=None  # 고정 전략이므로 학습자 없음
            )
            sta.decision_log = decision_log
            sta.current_episode = episode
            stas.append(sta)
        
        # 고정 전략 오버라이드
        action_override = 1 if strategy == "always_npca" else 0
        for sta in stas:
            sta._fixed_action = action_override
        
        # 시뮬레이터 생성 및 실행
        simulator = Simulator(num_slots=200, channels=channels, stas=stas)
        simulator.device = device
        simulator.run()
        
        # 에피소드 보상 수집
        total_reward = sum(sta.episode_reward for sta in stas)
        episode_rewards.append(total_reward / 100)  # 정규화
    
    return episode_rewards, decision_log

def plot_comparison(results, save_dir="./comparison_results"):
    """비교 결과 플롯"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 색상 설정
    colors = {'DRL Policy': 'blue', 'Always NPCA': 'red', 'Always Primary': 'green'}
    
    # 1. 에피소드별 보상
    plt.subplot(2, 2, 1)
    for name, data in results.items():
        episodes = range(len(data['rewards']))
        plt.plot(episodes, data['rewards'], label=name, alpha=0.7, 
                color=colors.get(name, 'black'))
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Running Average
    plt.subplot(2, 2, 2)
    window = 20
    for name, data in results.items():
        rewards = data['rewards']
        running_avg = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]
        plt.plot(running_avg, label=name, linewidth=2, color=colors.get(name, 'black'))
    plt.title(f'Running Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box Plot
    plt.subplot(2, 2, 3)
    data_list = [results[name]['rewards'] for name in results.keys()]
    labels = list(results.keys())
    bp = plt.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, 'lightblue'))
        patch.set_alpha(0.7)
    plt.title('Reward Distribution')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # 4. Performance Bar Chart
    plt.subplot(2, 2, 4)
    names = list(results.keys())
    means = [np.mean(results[name]['rewards']) for name in names]
    stds = [np.std(results[name]['rewards']) for name in names]
    
    bars = plt.bar(names, means, yerr=stds, capsize=5,
                  color=[colors.get(name, 'lightblue') for name in names],
                  alpha=0.7, edgecolor='black')
    plt.title('Average Performance')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {save_dir}/strategy_comparison.png")

def main():
    """메인 비교 함수"""
    print("="*60)
    print("DRL vs Baseline 전략 성능 비교")
    print("="*60)
    
    # 설정
    channels, stas_config = create_test_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    num_episodes = 50  # 테스트용
    
    print(f"Device: {device}")
    print(f"Test episodes: {num_episodes}")
    print(f"Model: {model_path}")
    print()
    
    results = {}
    
    # 1. DRL Policy 테스트
    drl_rewards, drl_log = test_drl_policy(channels, stas_config, model_path, num_episodes, device)
    if drl_rewards:
        results['DRL Policy'] = {'rewards': drl_rewards, 'decisions': drl_log}
    
    # 2. Always NPCA 테스트
    npca_rewards, npca_log = test_fixed_strategy(channels, stas_config, "always_npca", num_episodes, device)
    results['Always NPCA'] = {'rewards': npca_rewards, 'decisions': npca_log}
    
    # 3. Always Primary 테스트  
    primary_rewards, primary_log = test_fixed_strategy(channels, stas_config, "always_primary", num_episodes, device)
    results['Always Primary'] = {'rewards': primary_rewards, 'decisions': primary_log}
    
    # 결과 저장
    results_dir = "./comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # CSV 저장
    df_data = {}
    for name, data in results.items():
        df_data[name] = data['rewards']
    
    pd.DataFrame(df_data).to_csv(f"{results_dir}/rewards_comparison.csv", index=False)
    
    # 플롯 생성
    plot_comparison(results, results_dir)
    
    # 결과 출력
    print("\n" + "="*60)
    print("성능 비교 결과")
    print("="*60)
    
    for name, data in results.items():
        rewards = data['rewards']
        print(f"{name:15} - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}, Max: {np.max(rewards):.3f}")
    
    # 상대 성능 계산
    if 'DRL Policy' in results and 'Always NPCA' in results:
        drl_mean = np.mean(results['DRL Policy']['rewards'])
        npca_mean = np.mean(results['Always NPCA']['rewards'])
        improvement = ((drl_mean - npca_mean) / npca_mean) * 100
        print(f"\nDRL vs Always NPCA 개선: {improvement:+.1f}%")
    
    if 'DRL Policy' in results and 'Always Primary' in results:
        drl_mean = np.mean(results['DRL Policy']['rewards'])
        primary_mean = np.mean(results['Always Primary']['rewards'])
        improvement = ((drl_mean - primary_mean) / primary_mean) * 100
        print(f"DRL vs Always Primary 개선: {improvement:+.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    main()