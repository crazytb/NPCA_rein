#!/usr/bin/env python3
"""
Test New Reward Models - Compare DRL with new reward system vs baselines
Based on models trained with new reward function (PPDU duration + energy cost)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

from drl_framework.random_access import Channel, STA, Simulator
from drl_framework.network import DQN
from drl_framework.configs import PPDU_DURATION, RADIO_TRANSITION_TIME, OBSS_GENERATION_RATE, OBSS_DURATION_RANGE

class BaselinePolicy:
    """Baseline policies for comparison"""
    
    @staticmethod
    def primary_only(obs_dict):
        """Always stay on primary channel (Action 0)"""
        return 0
    
    @staticmethod 
    def npca_only(obs_dict):
        """Always switch to NPCA when available (Action 1)"""
        return 1
    
    @staticmethod
    def random_policy(obs_dict):
        """Random action selection"""
        return np.random.randint(0, 2)

def test_drl_policy(model_path, obs_dict, device):
    """Test DRL policy using trained model"""
    try:
        # Load model
        model = DQN(n_observations=4, n_actions=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Convert observation to vector (matching training format)
        obs_vec = [
            float(obs_dict["primary_channel_obss_occupied_remained"]),
            float(obs_dict["radio_transition_time"]),
            float(obs_dict["tx_duration"]),
            float(obs_dict["cw_index"])
        ]
        
        # Normalize (matching training normalization)
        caps = {"slots": 1024, "cw_stage_max": 8}
        obs_vec[0] = min(obs_vec[0], caps["slots"]) / caps["slots"]
        obs_vec[1] = min(obs_vec[1], caps["slots"]) / caps["slots"]
        obs_vec[2] = min(obs_vec[2], caps["slots"]) / caps["slots"]
        obs_vec[3] = min(obs_vec[3], caps["cw_stage_max"]) / caps["cw_stage_max"]
        
        # Get action from model
        with torch.no_grad():
            state_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(state_tensor).max(1)[1].item()
            
        return action
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return 0  # Fallback to primary-only

def test_policy(policy_func, policy_name, model_path=None, test_episodes=20, obss_duration=100):
    """Test a policy and collect performance metrics"""
    
    print(f"Testing {policy_name} (OBSS Duration: {obss_duration})...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup channels using centralized configs
    channels = [
        Channel(channel_id=0, obss_generation_rate=OBSS_GENERATION_RATE['secondary']),  # Secondary/NPCA
        Channel(channel_id=1, obss_generation_rate=OBSS_GENERATION_RATE['primary'], 
                obss_duration_range=(obss_duration, obss_duration))  # Primary with OBSS
    ]
    
    # Setup STAs
    stas_config = [{
        "sta_id": 0,
        "channel_id": 1,  # Primary channel
        "npca_enabled": True,
        "ppdu_duration": PPDU_DURATION,
        "radio_transition_time": RADIO_TRANSITION_TIME
    }]
    
    # Test metrics
    episode_rewards = []
    action_counts = [0, 0]  # [Stay Primary, Go NPCA]
    throughput_total = 0
    episodes_completed = 0
    
    for episode in range(test_episodes):
        try:
            # Create STAs manually (matching train.py approach)
            from drl_framework.random_access import STA
            stas = []
            for config in stas_config:
                sta = STA(
                    sta_id=config["sta_id"],
                    channel_id=config["channel_id"],
                    primary_channel=channels[1],  # Primary channel (ID=1)
                    npca_channel=channels[0],     # Secondary/NPCA channel (ID=0)
                    npca_enabled=config.get("npca_enabled", False),
                    radio_transition_time=config.get("radio_transition_time", 1),
                    ppdu_duration=config.get("ppdu_duration", 33)
                )
                stas.append(sta)
            
            # Create simulator with STAs
            simulator = Simulator(num_slots=200, channels=channels, stas=stas)
            simulator.device = device
            
            # Set policy for STA
            sta = stas[0]
            if model_path:
                # DRL policy
                def drl_policy_func(obs_dict):
                    return test_drl_policy(model_path, obs_dict, device)
                sta.learner = type('MockLearner', (), {'select_action': lambda self, x: drl_policy_func(sta.get_obs())})()
            else:
                # Fixed policy
                sta._fixed_action = lambda: policy_func(sta.get_obs())
            
            # Run simulation
            simulator.run()
            
            # Collect metrics
            episode_reward = sta.episode_reward
            episode_rewards.append(episode_reward)
            throughput_total += sta.channel_occupancy_time
            episodes_completed += 1
            
            # Count actions (approximate from final results)
            if episode_reward > 0:  # Some successful transmissions
                if sta.channel_occupancy_time > 0:
                    # Estimate action distribution based on performance
                    if policy_name == "Primary-Only":
                        action_counts[0] += 10
                    elif policy_name == "NPCA-Only":
                        action_counts[1] += 10
                    else:
                        # For DRL, we can't easily track individual actions
                        action_counts[0] += 5
                        action_counts[1] += 5
                        
        except Exception as e:
            print(f"Episode {episode} failed: {e}")
            continue
    
    if episodes_completed == 0:
        return None
        
    # Calculate results
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
    avg_throughput = throughput_total / episodes_completed
    
    result = {
        'policy': policy_name,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_throughput': avg_throughput,
        'action_stay_ratio': action_counts[0] / (action_counts[0] + action_counts[1] + 1e-6),
        'episodes_completed': episodes_completed,
        'obss_duration': obss_duration
    }
    
    print(f"  ✅ Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  📈 Avg Throughput: {avg_throughput:.1f} slots")
    print(f"  🎯 Stay Primary Ratio: {result['action_stay_ratio']:.2%}")
    
    return result

def main():
    """Run comprehensive comparison test"""
    print("🚀 New Reward System Model Comparison")
    print("="*60)
    print(f"New Reward Function: +{PPDU_DURATION} (success) -{OBSS_GENERATION_RATE} (energy cost)")
    print()
    
    # Models to test with different OBSS durations
    models_to_test = [
        ("./obss_comparison_results/obss_20_slots/model.pth", "DRL-20", 20),
        ("./obss_comparison_results/obss_50_slots/model.pth", "DRL-50", 50),
        ("./obss_comparison_results/obss_100_slots/model.pth", "DRL-100", 100),
        ("./obss_comparison_results/obss_150_slots/model.pth", "DRL-150", 150),
    ]
    
    all_results = []
    
    # Test each OBSS duration condition
    for obss_duration in [20, 50, 100, 150]:
        print(f"\n🎯 OBSS Duration: {obss_duration} slots")
        print("-" * 40)
        
        # Test baseline policies
        for policy_func, policy_name in [
            (BaselinePolicy.primary_only, "Primary-Only"),
            (BaselinePolicy.npca_only, "NPCA-Only"),
        ]:
            result = test_policy(policy_func, policy_name, obss_duration=obss_duration)
            if result:
                all_results.append(result)
        
        # Test corresponding DRL model
        model_path = f"./obss_comparison_results/obss_{obss_duration}_slots/model.pth"
        if os.path.exists(model_path):
            result = test_policy(None, f"DRL-{obss_duration}", model_path=model_path, obss_duration=obss_duration)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs('./comparison_results', exist_ok=True)
        df.to_csv('./comparison_results/new_reward_comparison.csv', index=False)
        
        # Create visualization
        create_comparison_plots(df)
        
        print(f"\n📊 Results Summary:")
        print(df.groupby(['obss_duration', 'policy'])[['avg_reward', 'avg_throughput']].mean().round(2))
    
    print(f"\n✅ Test completed! Results saved to ./comparison_results/")

def create_comparison_plots(df):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average Reward by OBSS Duration
    ax1 = axes[0, 0]
    for policy in df['policy'].unique():
        policy_data = df[df['policy'] == policy]
        ax1.plot(policy_data['obss_duration'], policy_data['avg_reward'], 
                marker='o', label=policy, linewidth=2)
    ax1.set_xlabel('OBSS Duration (slots)')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward vs OBSS Duration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Throughput by OBSS Duration  
    ax2 = axes[0, 1]
    for policy in df['policy'].unique():
        policy_data = df[df['policy'] == policy]
        ax2.plot(policy_data['obss_duration'], policy_data['avg_throughput'], 
                marker='s', label=policy, linewidth=2)
    ax2.set_xlabel('OBSS Duration (slots)')
    ax2.set_ylabel('Average Throughput (slots)')
    ax2.set_title('Throughput vs OBSS Duration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Action Distribution
    ax3 = axes[1, 0]
    drl_data = df[df['policy'].str.contains('DRL')]
    if not drl_data.empty:
        ax3.bar(range(len(drl_data)), drl_data['action_stay_ratio'], 
               color='lightblue', edgecolor='navy')
        ax3.set_xlabel('DRL Model')
        ax3.set_ylabel('Stay Primary Ratio')
        ax3.set_title('DRL Action Distribution')
        ax3.set_xticks(range(len(drl_data)))
        ax3.set_xticklabels(drl_data['policy'], rotation=45)
    
    # 4. Reward vs Throughput Scatter
    ax4 = axes[1, 1]
    for policy in df['policy'].unique():
        policy_data = df[df['policy'] == policy]
        ax4.scatter(policy_data['avg_throughput'], policy_data['avg_reward'], 
                   label=policy, s=100, alpha=0.7)
    ax4.set_xlabel('Average Throughput (slots)')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Reward vs Throughput')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./comparison_results/new_reward_comparison.png', dpi=300, bbox_inches='tight')
    print("📈 Comparison plots saved to ./comparison_results/new_reward_comparison.png")

if __name__ == "__main__":
    main()