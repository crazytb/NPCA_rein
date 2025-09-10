#!/usr/bin/env python3
"""
Test Balanced DRL Model
Compare old vs new DRL models
"""

import torch
import numpy as np
import os
from comparison_test import test_policy, BaselinePolicy, dict_to_tensor, analyze_action_rewards
from drl_framework.network import DQN

def test_balanced_model():
    """Test the new balanced DRL model"""
    print("🧪 Testing Balanced DRL Model...")
    print("="*60)
    
    # Test conditions
    obss_duration = 150
    obss_rate = 0.03
    test_episodes = 50
    
    print(f"Test Conditions: OBSS Duration={obss_duration}, Rate={obss_rate}, Episodes={test_episodes}")
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
    
    # Test old Enhanced DRL model
    old_model_path = "./enhanced_results/enhanced_drl_model.pth"
    if os.path.exists(old_model_path):
        try:
            print("Loading Old Enhanced DRL model...")
            device = torch.device("cpu")
            old_policy_net = DQN(n_actions=2, history_length=10).to(device)
            checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)
            old_policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            old_policy_net.eval()
            
            old_result = test_policy(
                old_policy_net, "Old Enhanced DRL",
                test_episodes=test_episodes,
                obss_duration=obss_duration,
                obss_rate=obss_rate
            )
            results.append(old_result)
        except Exception as e:
            print(f"Error loading old model: {e}")
    
    # Test new Balanced DRL model
    new_model_path = "./balanced_results/balanced_drl_model.pth"
    if os.path.exists(new_model_path):
        try:
            print("Loading New Balanced DRL model...")
            device = torch.device("cpu")
            new_policy_net = DQN(n_actions=2, history_length=10).to(device)
            checkpoint = torch.load(new_model_path, map_location=device, weights_only=False)
            new_policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            new_policy_net.eval()
            
            new_result = test_policy(
                new_policy_net, "New Balanced DRL",
                test_episodes=test_episodes,
                obss_duration=obss_duration,
                obss_rate=obss_rate
            )
            results.append(new_result)
        except Exception as e:
            print(f"Error loading new model: {e}")
    
    # Analyze results
    analyze_action_rewards(results)
    
    # Compare DRL models
    print(f"\n{'🆚 DRL MODEL COMPARISON:'}")
    print("="*60)
    
    old_drl = None
    new_drl = None
    
    for result in results:
        if result['policy_name'] == 'Old Enhanced DRL':
            old_drl = result
        elif result['policy_name'] == 'New Balanced DRL':
            new_drl = result
    
    if old_drl and new_drl:
        print(f"{'Metric':<20} {'Old DRL':<15} {'New DRL':<15} {'Improvement'}")
        print("-" * 65)
        
        reward_diff = new_drl['avg_reward'] - old_drl['avg_reward']
        efficiency_diff = new_drl['efficiency'] - old_drl['efficiency']
        
        print(f"{'Avg Reward':<20} {old_drl['avg_reward']:>10.1f}   {new_drl['avg_reward']:>10.1f}   {reward_diff:>+8.1f}")
        print(f"{'Efficiency':<20} {old_drl['efficiency']:>10.3f}   {new_drl['efficiency']:>10.3f}   {efficiency_diff:>+8.3f}")
        print(f"{'Stay Action %':<20} {old_drl['action_distribution'][0]*100:>10.1f}   {new_drl['action_distribution'][0]*100:>10.1f}   {(new_drl['action_distribution'][0]-old_drl['action_distribution'][0])*100:>+8.1f}")
        print(f"{'Switch Action %':<20} {old_drl['action_distribution'][1]*100:>10.1f}   {new_drl['action_distribution'][1]*100:>10.1f}   {(new_drl['action_distribution'][1]-old_drl['action_distribution'][1])*100:>+8.1f}")
        
        print(f"\nSummary:")
        if reward_diff > 0:
            print(f"✅ New model performs {reward_diff:.1f} points better!")
        else:
            print(f"❌ New model performs {-reward_diff:.1f} points worse")
            
        if new_drl['action_distribution'][1] > 0.1:
            print(f"✅ New model uses both actions (Switch: {new_drl['action_distribution'][1]*100:.1f}%)")
        else:
            print(f"❌ New model still only uses Stay action")
    
    # Overall ranking
    print(f"\n{'🏆 OVERALL RANKING:'}")
    print("="*60)
    
    sorted_results = sorted(results, key=lambda x: x['avg_reward'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        stay_pct = result['action_distribution'][0] * 100
        switch_pct = result['action_distribution'][1] * 100
        print(f"{i+1}. {result['policy_name']:<18}: {result['avg_reward']:6.1f} ± {result['std_reward']:5.1f} "
              f"(Stay: {stay_pct:3.0f}%, Switch: {switch_pct:3.0f}%, Eff: {result['efficiency']:.3f})")

if __name__ == "__main__":
    test_balanced_model()