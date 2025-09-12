#!/usr/bin/env python3
"""
Test Simplified Reward Function
"""

import numpy as np
from comparison_test import test_policy, BaselinePolicy, analyze_action_rewards

def test_simplified_reward():
    """Test the simplified reward function"""
    print("🧪 Testing Simplified Reward Function...")
    print("="*60)
    
    # Test conditions
    obss_duration = 150
    obss_rate = 0.03
    test_episodes = 30
    
    print(f"Test Conditions: OBSS Duration={obss_duration}, Rate={obss_rate}")
    print("Simplified Reward: Throughput - Latency")
    print()
    
    results = []
    
    # Test baseline policies
    for policy_func, policy_name in [
        (BaselinePolicy.primary_only, "Primary-Only"),
        (BaselinePolicy.npca_only, "NPCA-Only"),
        (BaselinePolicy.random_policy, "Random")
    ]:
        result = test_policy(
            policy_func, policy_name, 
            test_episodes=test_episodes, 
            obss_duration=obss_duration,
            obss_rate=obss_rate
        )
        results.append(result)
    
    # Analyze results with detailed reward breakdown
    analyze_action_rewards(results)
    
    # Detailed analysis of reward components
    print(f"\n{'📊 REWARD COMPONENT ANALYSIS:'}")
    print("="*60)
    
    for result in results:
        policy_name = result['policy_name']
        detailed_metrics = result['detailed_metrics']
        
        print(f"\n{policy_name}:")
        
        # Collect reward components by action
        stay_throughput = []
        stay_latency = []
        stay_total = []
        switch_throughput = []
        switch_latency = []
        switch_total = []
        
        for episode_metrics in detailed_metrics:
            for action_data in episode_metrics['actions']:
                if action_data['action'] == 0:  # Stay Primary
                    stay_throughput.append(action_data['throughput'])
                    stay_latency.append(action_data['duration'])
                    stay_total.append(action_data['reward'])
                else:  # Switch NPCA
                    switch_throughput.append(action_data['throughput'])
                    switch_latency.append(action_data['duration'])
                    switch_total.append(action_data['reward'])
        
        if stay_throughput:
            print(f"  Stay Primary ({len(stay_throughput)} actions):")
            print(f"    Avg Throughput: {np.mean(stay_throughput):6.2f} ± {np.std(stay_throughput):5.2f}")
            print(f"    Avg Duration:   {np.mean(stay_latency):6.2f} ± {np.std(stay_latency):5.2f}")
            print(f"    Avg Reward:     {np.mean(stay_total):6.2f} ± {np.std(stay_total):5.2f}")
            
        if switch_throughput:
            print(f"  Switch NPCA ({len(switch_throughput)} actions):")
            print(f"    Avg Throughput: {np.mean(switch_throughput):6.2f} ± {np.std(switch_throughput):5.2f}")
            print(f"    Avg Duration:   {np.mean(switch_latency):6.2f} ± {np.std(switch_latency):5.2f}")
            print(f"    Avg Reward:     {np.mean(switch_total):6.2f} ± {np.std(switch_total):5.2f}")
        
        # Compare rewards if both actions exist
        if stay_total and switch_total:
            reward_diff = np.mean(stay_total) - np.mean(switch_total)
            throughput_diff = np.mean(stay_throughput) - np.mean(switch_throughput)
            latency_diff = np.mean(stay_latency) - np.mean(switch_latency)
            
            print(f"  Comparison (Stay - Switch):")
            print(f"    Reward Difference:     {reward_diff:+7.2f}")
            print(f"    Throughput Difference: {throughput_diff:+7.2f}")
            print(f"    Duration Difference:   {latency_diff:+7.2f}")
            
            if reward_diff > 0:
                print(f"    → Stay Primary is better due to:")
                if throughput_diff > 0:
                    print(f"      • Higher throughput (+{throughput_diff:.2f})")
                if latency_diff < 0:
                    print(f"      • Lower duration ({latency_diff:.2f})")
            else:
                print(f"    → Switch NPCA is better due to:")
                if throughput_diff < 0:
                    print(f"      • Higher throughput (+{-throughput_diff:.2f})")
                if latency_diff > 0:
                    print(f"      • Lower duration ({-latency_diff:.2f})")
    
    # Overall ranking
    print(f"\n{'🏆 RANKING (Simplified Reward):'}")
    print("="*60)
    
    sorted_results = sorted(results, key=lambda x: x['avg_reward'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        stay_pct = result['action_distribution'][0] * 100
        switch_pct = result['action_distribution'][1] * 100
        print(f"{i+1}. {result['policy_name']:<15}: {result['avg_reward']:6.1f} ± {result['std_reward']:5.1f} "
              f"(Stay: {stay_pct:3.0f}%, Switch: {switch_pct:3.0f}%)")

if __name__ == "__main__":
    test_simplified_reward()