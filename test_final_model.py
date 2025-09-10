#!/usr/bin/env python3
"""
Test Final Simplified DRL Model
Compare all models with simplified reward
"""

import torch
import numpy as np
import os
from comparison_test import test_policy, BaselinePolicy, dict_to_tensor, analyze_action_rewards
from drl_framework.network import DQN

def test_final_model():
    """Test the final simplified DRL model"""
    print("🎯 Final Model Comparison with Simplified Reward")
    print("="*60)
    
    # Test conditions
    obss_duration = 150
    obss_rate = 0.03
    test_episodes = 50
    
    print(f"Test Conditions: OBSS Duration={obss_duration}, Rate={obss_rate}")
    print(f"Reward Function: Throughput (weight=10.0) - Latency (weight=0.1)")
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
    
    # Test all DRL models
    models_to_test = [
        ("./enhanced_results/enhanced_drl_model.pth", "Old Enhanced DRL"),
        ("./balanced_results/balanced_drl_model.pth", "Balanced DRL"),
        ("./simplified_results/simplified_drl_model.pth", "Simplified DRL")
    ]
    
    for model_path, model_name in models_to_test:
        if os.path.exists(model_path):
            try:
                print(f"Loading {model_name}...")
                device = torch.device("cpu")
                policy_net = DQN(n_actions=2, history_length=10).to(device)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                policy_net.eval()
                
                result = test_policy(
                    policy_net, model_name,
                    test_episodes=test_episodes,
                    obss_duration=obss_duration,
                    obss_rate=obss_rate
                )
                results.append(result)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"{model_name} not found, skipping...")
    
    # Analyze results
    analyze_action_rewards(results)
    
    # DRL model comparison
    print(f"\n{'🤖 DRL MODEL EVOLUTION:'}")
    print("="*80)
    
    drl_models = {}
    for result in results:
        if 'DRL' in result['policy_name']:
            drl_models[result['policy_name']] = result
    
    if drl_models:
        print(f"{'Model':<18} {'Reward':<10} {'Stay%':<8} {'Switch%':<8} {'Efficiency'}")
        print("-" * 60)
        
        for model_name, result in drl_models.items():
            stay_pct = result['action_distribution'][0] * 100
            switch_pct = result['action_distribution'][1] * 100
            print(f"{model_name:<18} {result['avg_reward']:>7.1f}   {stay_pct:>5.1f}%   {switch_pct:>6.1f}%   {result['efficiency']:>8.3f}")
    
    # Final ranking
    print(f"\n{'🏆 FINAL RANKING:'}")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda x: x['avg_reward'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        stay_pct = result['action_distribution'][0] * 100
        switch_pct = result['action_distribution'][1] * 100
        print(f"{i+1}. {result['policy_name']:<18}: {result['avg_reward']:6.1f} ± {result['std_reward']:5.1f} "
              f"(Stay: {stay_pct:3.0f}%, Switch: {switch_pct:3.0f}%, Eff: {result['efficiency']:.3f})")
    
    # Success analysis
    print(f"\n{'📈 SUCCESS ANALYSIS:'}")
    print("="*60)
    
    # Find models that use both actions
    balanced_models = [r for r in results if r['action_distribution'][1] > 0.1]  # Switch > 10%
    
    if balanced_models:
        print("✅ Models using both actions:")
        for result in balanced_models:
            switch_pct = result['action_distribution'][1] * 100
            print(f"  • {result['policy_name']}: {switch_pct:.1f}% Switch, Reward: {result['avg_reward']:.1f}")
    
    # Find best performing model
    best_model = sorted_results[0]
    print(f"\n🎖️  Best Model: {best_model['policy_name']}")
    print(f"   Reward: {best_model['avg_reward']:.1f} ± {best_model['std_reward']:.1f}")
    print(f"   Actions: {best_model['action_distribution'][0]*100:.0f}% Stay, {best_model['action_distribution'][1]*100:.0f}% Switch")
    
    # Check if DRL learned the right policy
    simplified_drl = None
    for result in results:
        if result['policy_name'] == 'Simplified DRL':
            simplified_drl = result
            break
    
    if simplified_drl:
        print(f"\n🧠 Simplified DRL Analysis:")
        print(f"   Switch Usage: {simplified_drl['action_distribution'][1]*100:.1f}%")
        
        if simplified_drl['action_distribution'][1] > 0.3:  # > 30% switch
            print("   ✅ Successfully learned to use both actions!")
        elif simplified_drl['action_distribution'][1] > 0.1:  # > 10% switch
            print("   ⚠️  Partially learned to use Switch action")
        else:
            print("   ❌ Still only uses Stay action")
            
        # Compare with Random (which we know is optimal)
        random_result = next((r for r in results if r['policy_name'] == 'Random'), None)
        if random_result:
            performance_gap = simplified_drl['avg_reward'] - random_result['avg_reward']
            print(f"   Performance vs Random: {performance_gap:+.1f}")
            if performance_gap > 0:
                print("   ✅ Outperforms Random policy!")
            else:
                print("   ❌ Underperforms Random policy")

if __name__ == "__main__":
    test_final_model()