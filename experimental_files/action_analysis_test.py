#!/usr/bin/env python3
"""
Action Reward Analysis Test
Detailed analysis of rewards by action type
"""

import torch
import numpy as np
from comparison_test import test_policy, BaselinePolicy, dict_to_tensor, analyze_action_rewards
from drl_framework.network import DQN
import os

def run_action_analysis():
    """Run detailed action analysis"""
    print("🔬 Starting Action Reward Analysis...")
    
    results = []
    
    # Test scenario with moderate interference
    obss_duration = 150
    obss_rate = 0.03
    
    print(f"Test Conditions: OBSS Duration={obss_duration}, Rate={obss_rate}")
    print("="*60)
    
    # Test Random policy (uses both actions)
    random_result = test_policy(
        BaselinePolicy.random_policy,
        "Random Policy",
        test_episodes=50,
        obss_duration=obss_duration,
        obss_rate=obss_rate
    )
    results.append(random_result)
    
    # Test Enhanced DRL
    model_path = "./enhanced_results/enhanced_drl_model.pth"
    if os.path.exists(model_path):
        try:
            device = torch.device("cpu")
            policy_net = DQN(n_actions=2, history_length=10).to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            policy_net.eval()
            
            drl_result = test_policy(
                policy_net,
                "Enhanced DRL",
                test_episodes=50,
                obss_duration=obss_duration,
                obss_rate=obss_rate
            )
            results.append(drl_result)
        except Exception as e:
            print(f"Error loading Enhanced DRL: {e}")
    
    # Analyze action rewards
    analyze_action_rewards(results)
    
    # Additional detailed analysis for Random policy
    random_detailed = random_result['detailed_metrics']
    
    print(f"\n{'📊 DETAILED RANDOM POLICY ANALYSIS:'}")
    print("="*60)
    
    stay_rewards = []
    switch_rewards = []
    stay_throughputs = []
    switch_throughputs = []
    stay_waiting = []
    switch_waiting = []
    
    for episode_metrics in random_detailed:
        for action_data in episode_metrics['actions']:
            if action_data['action'] == 0:  # Stay Primary
                stay_rewards.append(action_data['reward'])
                stay_throughputs.append(action_data['throughput'])
                stay_waiting.append(action_data['waiting_time'])
            else:  # Switch NPCA
                switch_rewards.append(action_data['reward'])
                switch_throughputs.append(action_data['throughput'])
                switch_waiting.append(action_data['waiting_time'])
    
    if stay_rewards and switch_rewards:
        print(f"\nStay Primary Actions ({len(stay_rewards)} total):")
        print(f"  Reward     : {np.mean(stay_rewards):7.2f} ± {np.std(stay_rewards):6.2f}")
        print(f"  Throughput : {np.mean(stay_throughputs):7.2f} ± {np.std(stay_throughputs):6.2f}")
        print(f"  Wait Time  : {np.mean(stay_waiting):7.2f} ± {np.std(stay_waiting):6.2f}")
        
        print(f"\nSwitch NPCA Actions ({len(switch_rewards)} total):")
        print(f"  Reward     : {np.mean(switch_rewards):7.2f} ± {np.std(switch_rewards):6.2f}")
        print(f"  Throughput : {np.mean(switch_throughputs):7.2f} ± {np.std(switch_throughputs):6.2f}")
        print(f"  Wait Time  : {np.mean(switch_waiting):7.2f} ± {np.std(switch_waiting):6.2f}")
        
        # Simple statistical analysis
        reward_diff = np.mean(stay_rewards) - np.mean(switch_rewards)
        throughput_diff = np.mean(stay_throughputs) - np.mean(switch_throughputs)
        waiting_diff = np.mean(switch_waiting) - np.mean(stay_waiting)  # Switch - Stay for waiting
        
        print(f"\nAction Comparison:")
        print(f"  Reward Advantage (Stay - Switch)     : {reward_diff:+7.2f}")
        print(f"  Throughput Advantage (Stay - Switch) : {throughput_diff:+7.2f}")
        print(f"  Extra Waiting Time (Switch - Stay)   : {waiting_diff:+7.2f}")
        
        if reward_diff > 50:
            print(f"  → Stay Primary provides substantially higher rewards")
        elif reward_diff > 0:
            print(f"  → Stay Primary provides moderately higher rewards")
        elif reward_diff < -50:
            print(f"  → Switch NPCA provides substantially higher rewards")
        elif reward_diff < 0:
            print(f"  → Switch NPCA provides moderately higher rewards")
        else:
            print(f"  → Actions provide similar rewards")

if __name__ == "__main__":
    run_action_analysis()