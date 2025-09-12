#!/usr/bin/env python3
"""
Train Balanced Semi-MDP Model for NPCA Decision Making
Improved version with better exploration and diverse training scenarios
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import math
import os

from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN, ReplayMemory
from drl_framework.random_access import Channel

# Training hyperparameters
BATCH_SIZE = 64
LR = 5e-4
GAMMA = 0.95
EPS_START = 1.0  # Start with full exploration
EPS_END = 0.02   # Higher final exploration
EPS_DECAY = 2000 # Slower decay for more exploration
TAU = 0.005
TARGET_UPDATE = 10
MEMORY_SIZE = 20000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

def dict_to_tensor(obs_dict, device):
    """Convert observation dictionary to tensor format"""
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

def create_diverse_environment(scenario_type="balanced"):
    """Create environment with diverse training scenarios"""
    
    if scenario_type == "low_interference":
        return NPCASemiMDPEnv(
            num_stas=2,
            num_slots=500,
            obss_generation_rate=0.01,
            npca_enabled=True,
            throughput_weight=10.0,
            latency_penalty_weight=0.1,
            history_length=10
        ), 50, 0.01, 0  # obss_duration, obss_rate, primary_obss_rate
        
    elif scenario_type == "high_interference":
        return NPCASemiMDPEnv(
            num_stas=2,
            num_slots=500,
            obss_generation_rate=0.05,
            npca_enabled=True,
            throughput_weight=10.0,
            latency_penalty_weight=0.1,
            history_length=10
        ), 200, 0.05, 0
        
    elif scenario_type == "mixed_interference":
        return NPCASemiMDPEnv(
            num_stas=2,
            num_slots=500,
            obss_generation_rate=0.03,
            npca_enabled=True,
            throughput_weight=10.0,
            latency_penalty_weight=0.1,
            history_length=10
        ), 150, 0.03, 0.01
        
    else:  # balanced
        return NPCASemiMDPEnv(
            num_stas=2,
            num_slots=500,
            obss_generation_rate=0.02,
            npca_enabled=True,
            throughput_weight=10.0,
            latency_penalty_weight=0.1,
            history_length=10
        ), 120, 0.025, 0

def setup_environment_channels(env, obss_duration, obss_rate, primary_obss_rate):
    """Setup environment with specified channel conditions"""
    channels = [
        Channel(channel_id=0, obss_generation_rate=primary_obss_rate),
        Channel(channel_id=1, obss_generation_rate=obss_rate, 
                obss_duration_range=(obss_duration, obss_duration))
    ]
    
    env.primary_channel = channels[0]
    env.npca_channel = channels[1]
    env.channels = channels
    
    for sta in env.stas:
        sta.primary_channel = channels[0]
        sta.npca_channel = channels[1]

def select_action(state, policy_net, steps_done, device, forced_exploration=False):
    """Select action with improved exploration strategy"""
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    # Force some exploration in early stages
    if forced_exploration and steps_done < 500:
        eps_threshold = max(eps_threshold, 0.5)
    
    if sample > eps_threshold:
        with torch.no_grad():
            state_tensor = dict_to_tensor(state, device)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].view(1, 1).item(), eps_threshold
    else:
        return random.randrange(2), eps_threshold

def optimize_model(memory, policy_net, target_net, optimizer, device):
    """Optimize the model with improved loss function"""
    if len(memory) < BATCH_SIZE:
        return None
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Process states and actions
    state_action_values = []
    next_state_values = []
    rewards = []
    
    for i in range(BATCH_SIZE):
        state = batch.state[i]
        action = batch.action[i]
        reward = batch.cum_reward[i]
        next_state = batch.next_state[i]
        
        # Current Q value
        state_tensor = dict_to_tensor(state, device)
        q_values = policy_net(state_tensor)
        state_action_values.append(q_values[0, action])
        
        # Next state Q value
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = dict_to_tensor(next_state, device)
                next_q_values = target_net(next_state_tensor)
                next_state_values.append(next_q_values.max().item())
        else:
            next_state_values.append(0.0)
            
        rewards.append(reward)
    
    # Convert to tensors
    state_action_values = torch.stack(state_action_values)
    expected_values = torch.tensor([r + GAMMA * nv for r, nv in zip(rewards, next_state_values)], 
                                   device=device, dtype=torch.float32)
    
    # Compute loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def train_balanced_model():
    """Train model with simplified reward and diverse scenarios"""
    
    print("🚀 Starting Simplified Reward DRL Training...")
    print("="*60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize networks
    policy_net = DQN(n_actions=2, history_length=10).to(device)
    target_net = DQN(n_actions=2, history_length=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Training tracking
    episode_rewards = []
    episode_losses = []
    action_counts = [0, 0]  # [Stay, Switch]
    steps_done = 0
    
    # Training scenarios rotation
    scenarios = ["balanced", "low_interference", "high_interference", "mixed_interference"]
    
    num_episodes = 200  # Reduced for faster training
    scenario_switch_freq = 25  # Change scenario every 25 episodes
    
    print("Training Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Memory size: {MEMORY_SIZE}")
    print(f"  Exploration: {EPS_START} → {EPS_END} (decay: {EPS_DECAY})")
    print(f"  Scenarios: {scenarios}")
    print()
    
    for episode in range(num_episodes):
        
        # Select scenario
        scenario_idx = (episode // scenario_switch_freq) % len(scenarios)
        current_scenario = scenarios[scenario_idx]
        
        # Create environment for current scenario
        env, obss_duration, obss_rate, primary_obss_rate = create_diverse_environment(current_scenario)
        setup_environment_channels(env, obss_duration, obss_rate, primary_obss_rate)
        
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = []
        decisions_made = 0
        eps_threshold = 1.0  # Initialize epsilon threshold
        
        # Episode loop
        while env.current_slot < env.num_slots:
            # Check if we're at a decision point
            if not env._is_decision_point(env.decision_sta, env.current_slot):
                if not env._advance_to_next_decision():
                    break
                state = env._get_observation()
                continue
            
            # Select action with forced exploration in early episodes
            forced_exploration = episode < 100
            action, eps_threshold = select_action(state, policy_net, steps_done, device, forced_exploration)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            decisions_made += 1
            action_counts[action] += 1
            steps_done += 1
            
            # Store transition
            if done:
                next_state = None
            
            memory.push(state, action, next_state, reward, info.get('duration', 1), done)
            
            # Optimize model
            if len(memory) >= BATCH_SIZE:
                loss = optimize_model(memory, policy_net, target_net, optimizer, device)
                if loss is not None:
                    episode_loss.append(loss)
            
            if done:
                break
                
            state = next_state
        
        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Record episode results
        episode_rewards.append(episode_reward)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        else:
            episode_losses.append(0)
        
        # Progress reporting
        if episode % 10 == 0:
            recent_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            action_ratio = action_counts[1] / max(sum(action_counts), 1)
            
            print(f"Episode {episode:3d} | Scenario: {current_scenario:15s} | "
                  f"Reward: {recent_reward:6.1f} | Switch%: {action_ratio:.2f} | "
                  f"ε: {eps_threshold:.3f} | Decisions: {decisions_made}")
    
    # Save results
    os.makedirs("./simplified_results", exist_ok=True)
    
    # Save model
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'action_counts': action_counts,
        'training_scenarios': scenarios,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'gamma': GAMMA,
            'eps_start': EPS_START,
            'eps_end': EPS_END,
            'eps_decay': EPS_DECAY,
            'memory_size': MEMORY_SIZE
        }
    }, './simplified_results/simplified_drl_model.pth')
    
    print(f"\n✅ Training completed!")
    print(f"Final action distribution: Stay={action_counts[0]}, Switch={action_counts[1]}")
    print(f"Switch action ratio: {action_counts[1]/sum(action_counts):.3f}")
    print(f"Model saved to ./simplified_results/simplified_drl_model.pth")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Balanced DRL Training Results', fontsize=16)
    
    # Episode rewards
    axes[0,0].plot(episode_rewards, alpha=0.6)
    if len(episode_rewards) >= 20:
        moving_avg = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
        axes[0,0].plot(range(19, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
    axes[0,0].set_title('Episode Rewards')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].grid(True)
    
    # Training loss
    axes[0,1].plot(episode_losses)
    axes[0,1].set_title('Training Loss')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].grid(True)
    
    # Action distribution over time
    window_size = 50
    switch_ratios = []
    for i in range(window_size, len(episode_rewards)):
        # This is a simplified version - would need to track actions per episode for exact calculation
        switch_ratios.append(action_counts[1] / sum(action_counts))
    
    axes[1,0].plot(range(window_size, len(episode_rewards)), switch_ratios)
    axes[1,0].set_title('Action Distribution (Switch Ratio)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Switch Action Ratio')
    axes[1,0].grid(True)
    
    # Final action counts
    axes[1,1].bar(['Stay Primary', 'Switch NPCA'], action_counts, 
                  color=['lightblue', 'lightcoral'])
    axes[1,1].set_title('Total Action Counts')
    axes[1,1].set_ylabel('Count')
    for i, count in enumerate(action_counts):
        axes[1,1].text(i, count + max(action_counts)*0.01, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('./simplified_results/simplified_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return episode_rewards, action_counts

if __name__ == "__main__":
    train_balanced_model()