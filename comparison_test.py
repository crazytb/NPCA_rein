#!/usr/bin/env python3
"""
Comparison Test: Enhanced DRL vs Primary-Only vs NPCA-Only
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN
from drl_framework.random_access import Channel

def dict_to_tensor(obs_dict, device):
    """Convert observation dictionary to tensor format (matching train_enhanced_model.py)"""
    tensor_dict = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.float32)
        else:
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.float32)
        
        # Add batch dimension
        if tensor_dict[key].dim() == 0:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
        elif tensor_dict[key].dim() == 1 and key not in ['primary_busy_history', 'obss_busy_history', 'npca_busy_history', 'obss_frequency', 'avg_obss_duration']:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
        elif tensor_dict[key].dim() == 1:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
    
    return tensor_dict

class BaselinePolicy:
    """Baseline policies for comparison"""
    
    @staticmethod
    def primary_only(obs_dict):
        """Always stay on primary channel"""
        return 0
    
    @staticmethod 
    def npca_only(obs_dict):
        """Always switch to NPCA when available"""
        return 1
    
    @staticmethod
    def random_policy(obs_dict):
        """Random action selection"""
        return np.random.randint(0, 2)

def test_policy(policy_func, policy_name, test_episodes=50, obss_duration=100, obss_rate=0.01, primary_obss_rate=0):
    """Test a policy and collect performance metrics"""
    
    print(f"Testing {policy_name}...")
    
    # Environment setup
    env = NPCASemiMDPEnv(
        num_stas=2,
        num_slots=500,
        obss_generation_rate=0.01,
        npca_enabled=True,
        throughput_weight=10.0,
        latency_penalty_weight=0.1,
        history_length=10
    )
    
    # Set OBSS conditions for both channels
    channels = [
        Channel(channel_id=0, obss_generation_rate=primary_obss_rate),  # Primary channel can have OBSS too
        Channel(channel_id=1, obss_generation_rate=obss_rate, obss_duration_range=(obss_duration, obss_duration))
    ]
    env.primary_channel = channels[0]
    env.npca_channel = channels[1]
    env.channels = channels
    for sta in env.stas:
        sta.primary_channel = channels[0]
        sta.npca_channel = channels[1]
    
    # Test metrics
    episode_rewards = []
    episode_decisions = []
    detailed_metrics = []
    action_counts = [0, 0]  # [Stay, Switch]
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        decisions_made = 0
        episode_metrics = {
            'total_throughput': 0,
            'total_latency': 0,
            'total_duration': 0,
            'actions': []
        }
        
        while env.current_slot < env.num_slots:
            if not env._is_decision_point(env.decision_sta, env.current_slot):
                if not env._advance_to_next_decision():
                    break
                obs = env._get_observation()
                continue
            
            # Get action from policy
            if hasattr(policy_func, '__call__') and not hasattr(policy_func, 'parameters'):
                # Baseline policy function
                action = policy_func(obs)
            else:
                # DRL policy network
                with torch.no_grad():
                    # Debug: print observation types
                    if episode == 0 and decisions_made == 0:
                        print(f"Debug obs: {[(k, type(v), v) for k, v in obs.items()]}")
                        
                    state_tensor = dict_to_tensor(obs, torch.device("cpu"))
                    
                    # Debug tensor shapes
                    if episode == 0 and decisions_made == 0:
                        for k, v in state_tensor.items():
                            print(f"Tensor {k}: shape {v.shape}")
                    
                    q_values = policy_func(state_tensor)
                    action = q_values.max(1)[1].item()
            
            next_obs, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            decisions_made += 1
            action_counts[action] += 1
            
            # Collect detailed metrics
            episode_metrics['actions'].append({
                'action': action,
                'reward': reward,
                'throughput': info['successful_transmission_slots'],
                'waiting_time': info['waiting_slots'],
                'duration': info['duration']
            })
            
            episode_metrics['total_throughput'] += info['successful_transmission_slots']
            episode_metrics['total_latency'] += info['waiting_slots']
            episode_metrics['total_duration'] += info['duration']
            
            if done:
                break
                
            obs = next_obs
        
        episode_rewards.append(episode_reward)
        episode_decisions.append(decisions_made)
        detailed_metrics.append(episode_metrics)
    
    # Calculate summary statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_decisions = np.mean(episode_decisions)
    
    total_actions = sum(action_counts)
    action_probs = [count/max(total_actions, 1) for count in action_counts]
    
    # Aggregate metrics
    total_throughput = sum([ep['total_throughput'] for ep in detailed_metrics])
    total_latency = sum([ep['total_latency'] for ep in detailed_metrics])
    total_duration = sum([ep['total_duration'] for ep in detailed_metrics])
    
    efficiency = total_throughput / max(total_duration, 1)
    
    results = {
        'policy_name': policy_name,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_decisions': avg_decisions,
        'action_distribution': action_probs,
        'total_throughput': total_throughput,
        'total_latency': total_latency,
        'efficiency': efficiency,
        'episode_rewards': episode_rewards,
        'detailed_metrics': detailed_metrics
    }
    
    print(f"  {policy_name}: Avg Reward = {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Action Dist: Stay={action_probs[0]:.2f}, Switch={action_probs[1]:.2f}")
    print(f"  Efficiency: {efficiency:.4f}")
    
    return results

def analyze_action_rewards(results):
    """Analyze reward breakdown by action type"""
    print(f"\n{'🔍 ACTION REWARD ANALYSIS:'}")
    print("="*80)
    
    for result in results:
        policy_name = result['policy_name']
        detailed_metrics = result['detailed_metrics']
        
        # Collect rewards by action
        stay_rewards = []
        switch_rewards = []
        
        for episode_metrics in detailed_metrics:
            for action_data in episode_metrics['actions']:
                if action_data['action'] == 0:  # Stay Primary
                    stay_rewards.append(action_data['reward'])
                else:  # Switch NPCA
                    switch_rewards.append(action_data['reward'])
        
        print(f"\n{policy_name}:")
        if stay_rewards:
            avg_stay = np.mean(stay_rewards)
            std_stay = np.std(stay_rewards)
            print(f"  Stay Primary  : {len(stay_rewards):4d} actions, Avg Reward: {avg_stay:7.2f} ± {std_stay:6.2f}")
        else:
            print(f"  Stay Primary  :    0 actions")
            
        if switch_rewards:
            avg_switch = np.mean(switch_rewards)
            std_switch = np.std(switch_rewards)
            print(f"  Switch NPCA   : {len(switch_rewards):4d} actions, Avg Reward: {avg_switch:7.2f} ± {std_switch:6.2f}")
        else:
            print(f"  Switch NPCA   :    0 actions")
            
        # Compare action rewards if both exist
        if stay_rewards and switch_rewards:
            reward_diff = np.mean(stay_rewards) - np.mean(switch_rewards)
            if reward_diff > 0:
                print(f"  → Stay Primary is better by {reward_diff:.2f} points")
            else:
                print(f"  → Switch NPCA is better by {-reward_diff:.2f} points")

def run_scenario_test(scenario_name, obss_duration, obss_rate, primary_obss_rate=0):
    """Run test for a specific scenario"""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"OBSS Duration: {obss_duration}, OBSS Rate: {obss_rate}, Primary OBSS: {primary_obss_rate}")
    print(f"{'='*60}")
    
    results = []
    
    # Test all policies with the same scenario parameters
    for policy_func, policy_name in [
        (BaselinePolicy.primary_only, "Primary-Only"),
        (BaselinePolicy.npca_only, "NPCA-Only"),
        (BaselinePolicy.random_policy, "Random")
    ]:
        result = test_policy(
            policy_func, policy_name, 
            test_episodes=30, 
            obss_duration=obss_duration,
            obss_rate=obss_rate,
            primary_obss_rate=primary_obss_rate
        )
        results.append(result)
    
    # Test Enhanced DRL if available
    model_path = "./enhanced_results/enhanced_drl_model.pth"
    if os.path.exists(model_path):
        try:
            device = torch.device("cpu")
            policy_net = DQN(n_actions=2, history_length=10).to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            policy_net.eval()
            
            drl_result = test_policy(
                policy_net, "Enhanced DRL",
                test_episodes=30,
                obss_duration=obss_duration,
                obss_rate=obss_rate,
                primary_obss_rate=primary_obss_rate
            )
            results.append(drl_result)
        except Exception as e:
            print(f"Error loading Enhanced DRL: {e}")
    
    # Analyze action rewards for this scenario
    analyze_action_rewards(results)
    
    return results

def run_comparison_test():
    """Run multiple scenario comparison tests"""
    all_scenario_results = {}
    
    # Scenario 1: Current (Low interference)
    all_scenario_results["Low Interference"] = run_scenario_test(
        "Low Interference (Current)", 
        obss_duration=100, obss_rate=0.01
    )
    
    # Scenario 2: High interference frequency
    all_scenario_results["High Frequency"] = run_scenario_test(
        "High Interference Frequency", 
        obss_duration=100, obss_rate=0.05
    )
    
    # Scenario 3: Long duration interference
    all_scenario_results["Long Duration"] = run_scenario_test(
        "Long Duration Interference", 
        obss_duration=300, obss_rate=0.02
    )
    
    # Scenario 4: Both channels interfered
    all_scenario_results["Both Interfered"] = run_scenario_test(
        "Both Channels Interfered", 
        obss_duration=150, obss_rate=0.03, primary_obss_rate=0.01
    )
    
    # Scenario 5: Extreme interference
    all_scenario_results["Extreme"] = run_scenario_test(
        "Extreme Interference", 
        obss_duration=200, obss_rate=0.08
    )
    
    return all_scenario_results

def visualize_comparison(results, save_dir="./comparison_results"):
    """Create comprehensive comparison visualizations"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for plotting
    policy_names = [r['policy_name'] for r in results]
    avg_rewards = [r['avg_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NPCA Policy Comparison Results', fontsize=16, fontweight='bold')
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    # 1. Average Rewards with Error Bars
    ax1 = axes[0, 0]
    bars1 = ax1.bar(policy_names, avg_rewards, yerr=std_rewards, capsize=5, color=colors[:len(results)])
    ax1.set_title('Average Episode Reward')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    for i, (avg, std) in enumerate(zip(avg_rewards, std_rewards)):
        ax1.text(i, avg + std + max(avg_rewards)*0.02, f'{avg:.1f}', ha='center', fontweight='bold')
    
    # 2. Efficiency Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(policy_names, efficiencies, color=colors[:len(results)])
    ax2.set_title('Channel Utilization Efficiency')
    ax2.set_ylabel('Throughput / Duration')
    ax2.grid(True, alpha=0.3)
    for i, eff in enumerate(efficiencies):
        ax2.text(i, eff + max(efficiencies)*0.02, f'{eff:.3f}', ha='center', fontweight='bold')
    
    # 3. Action Distribution
    ax3 = axes[0, 2]
    stay_probs = [r['action_distribution'][0] for r in results]
    switch_probs = [r['action_distribution'][1] for r in results]
    
    x_pos = np.arange(len(policy_names))
    width = 0.35
    
    ax3.bar(x_pos - width/2, stay_probs, width, label='Stay Primary', color='lightblue', alpha=0.8)
    ax3.bar(x_pos + width/2, switch_probs, width, label='Switch NPCA', color='lightsalmon', alpha=0.8)
    
    ax3.set_title('Action Distribution')
    ax3.set_ylabel('Action Probability')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(policy_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward Distribution (Box Plot)
    ax4 = axes[1, 0]
    reward_data = [r['episode_rewards'] for r in results]
    box_plot = ax4.boxplot(reward_data, labels=policy_names, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_title('Reward Distribution')
    ax4.set_ylabel('Episode Reward')
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Table
    ax5 = axes[1, 1]
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            r['policy_name'],
            f"{r['avg_reward']:.1f}",
            f"{r['efficiency']:.3f}",
            f"{r['avg_decisions']:.1f}",
            f"{r['total_throughput']}"
        ])
    
    table = ax5.table(
        cellText=table_data,
        colLabels=['Policy', 'Avg Reward', 'Efficiency', 'Decisions', 'Total Throughput'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax5.set_title('Performance Summary', pad=20)
    
    # 6. Learning Curve (if DRL model exists)
    ax6 = axes[1, 2]
    drl_result = None
    for r in results:
        if r['policy_name'] == 'Enhanced DRL':
            drl_result = r
            break
    
    if drl_result and os.path.exists("./enhanced_results/enhanced_drl_model.pth"):
        checkpoint = torch.load("./enhanced_results/enhanced_drl_model.pth", map_location='cpu', weights_only=False)
        training_rewards = checkpoint['episode_rewards']
        
        ax6.plot(training_rewards, alpha=0.6, label='Episode Reward')
        
        # Moving average
        if len(training_rewards) >= 20:
            moving_avg = np.convolve(training_rewards, np.ones(20)/20, mode='valid')
            ax6.plot(range(19, len(training_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
        
        ax6.set_title('DRL Training Progress')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Reward')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'DRL Training\nData Not Available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('DRL Training Progress')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/policy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    df_data = []
    for r in results:
        df_data.append({
            'Policy': r['policy_name'],
            'Avg_Reward': r['avg_reward'],
            'Std_Reward': r['std_reward'],
            'Efficiency': r['efficiency'],
            'Stay_Prob': r['action_distribution'][0],
            'Switch_Prob': r['action_distribution'][1],
            'Avg_Decisions': r['avg_decisions'],
            'Total_Throughput': r['total_throughput'],
            'Total_Latency': r['total_latency']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{save_dir}/comparison_results.csv", index=False)
    
    print(f"\nResults saved to {save_dir}/")
    print("- policy_comparison.png")
    print("- comparison_results.csv")
    
    return results

def print_scenario_summary(all_scenario_results):
    """Print comprehensive summary across all scenarios"""
    print("\n" + "="*80)
    print("MULTI-SCENARIO COMPARISON SUMMARY")
    print("="*80)
    
    # Collect DRL action distributions across scenarios
    drl_actions = {}
    
    for scenario_name, results in all_scenario_results.items():
        print(f"\n{scenario_name}:")
        sorted_results = sorted(results, key=lambda x: x['avg_reward'], reverse=True)
        
        for i, r in enumerate(sorted_results):
            stay_pct = r['action_distribution'][0] * 100
            switch_pct = r['action_distribution'][1] * 100
            print(f"  {i+1}. {r['policy_name']:12s}: {r['avg_reward']:6.1f} ± {r['std_reward']:5.1f} "
                  f"(Stay: {stay_pct:3.0f}%, Switch: {switch_pct:3.0f}%, Eff: {r['efficiency']:.3f})")
            
            # Track DRL behavior
            if r['policy_name'] == 'Enhanced DRL':
                drl_actions[scenario_name] = {
                    'stay': stay_pct, 'switch': switch_pct, 
                    'reward': r['avg_reward'], 'efficiency': r['efficiency']
                }
    
    # DRL behavior analysis
    if drl_actions:
        print(f"\n{'🤖 DRL BEHAVIOR ANALYSIS:'}")
        print(f"{'Scenario':<20} {'Stay%':<8} {'Switch%':<8} {'Reward':<8} {'Efficiency'}")
        print("-" * 60)
        for scenario, actions in drl_actions.items():
            print(f"{scenario:<20} {actions['stay']:5.0f}%   {actions['switch']:5.0f}%   "
                  f"{actions['reward']:6.1f}   {actions['efficiency']:.3f}")

def main():
    """Main execution"""
    print("🚀 Starting multi-scenario policy comparison...")
    
    # Run comparison test
    all_scenario_results = run_comparison_test()
    
    # Print comprehensive summary
    print_scenario_summary(all_scenario_results)
    
    print("\n🎯 Multi-scenario comparison complete!")

if __name__ == "__main__":
    main()