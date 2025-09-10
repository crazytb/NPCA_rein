#!/usr/bin/env python3
"""
Enhanced Semi-MDP NPCA Training Test
- Multi-component reward system
- Channel history features 
- Improved network architecture
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import math

from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN, ReplayMemory
from drl_framework.params import *

# Ensure TARGET_UPDATE is defined
if 'TARGET_UPDATE' not in globals():
    TARGET_UPDATE = 100

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

def dict_to_tensor(obs_dict, device):
    """Convert observation dictionary to tensor format for network input"""
    # Convert each field to tensor and move to device
    tensor_dict = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.float32)
        else:
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.float32)
        
        # Add batch dimension if needed
        if tensor_dict[key].dim() == 0:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
        elif tensor_dict[key].dim() == 1 and key not in ['primary_busy_history', 'obss_busy_history', 'npca_busy_history', 'obss_frequency', 'avg_obss_duration']:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
        elif tensor_dict[key].dim() == 1 and key in ['primary_busy_history', 'obss_busy_history', 'npca_busy_history', 'obss_frequency', 'avg_obss_duration']:
            tensor_dict[key] = tensor_dict[key].unsqueeze(0)
    
    return tensor_dict

def select_action(obs_dict, policy_net, steps_done, device):
    """Enhanced epsilon-greedy action selection"""
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    if random.random() > eps_threshold:
        with torch.no_grad():
            state_tensor = dict_to_tensor(obs_dict, device)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()
    else:
        return random.randint(0, 1)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    """Enhanced optimization for dictionary states"""
    if len(memory) < BATCH_SIZE:
        return None
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Process state dictionaries for batching
    state_batch = {}
    next_state_batch = {}
    
    # Get all keys from first state
    sample_state = batch.state[0]
    for key in sample_state.keys():
        state_batch[key] = torch.stack([dict_to_tensor(s, device)[key].squeeze(0) for s in batch.state])
        # Only process next states that are not None (not terminal)
        next_states = [dict_to_tensor(s, device)[key].squeeze(0) for s in batch.next_state if s is not None]
        if next_states:
            next_state_batch[key] = torch.stack(next_states)

    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    reward_batch = torch.tensor(batch.cum_reward, device=device).float()
    tau_batch = torch.tensor(batch.tau, device=device).float()
    done_batch = torch.tensor(batch.done, device=device).float()

    # Current Q values
    current_q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Next Q values for non-terminal states
    next_q_values = torch.zeros(len(batch.state), device=device)
    non_final_mask = (done_batch == 0)
    
    if non_final_mask.sum() > 0 and next_state_batch:
        with torch.no_grad():
            next_q_values[non_final_mask] = target_net(next_state_batch).max(1)[0]

    # Semi-MDP TD target with tau discounting
    tau_clipped = torch.clamp(tau_batch, max=20.0)
    expected_q_values = reward_batch + (GAMMA ** tau_clipped) * next_q_values * (1.0 - done_batch)

    # Compute loss
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

def update_target_network(policy_net, target_net):
    """Soft update of target network"""
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                     target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

def train_enhanced_semi_mdp(num_episodes=200, obss_duration=100):
    """Enhanced training with multi-component rewards and channel history"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment setup with enhanced features
    # First create channels with fixed OBSS duration
    from drl_framework.random_access import Channel
    
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),  # Primary channel (no OBSS)
        Channel(channel_id=1, obss_generation_rate=0.01, obss_duration_range=(obss_duration, obss_duration))  # Fixed OBSS duration
    ]
    
    env = NPCASemiMDPEnv(
        num_stas=2, 
        num_slots=1000,
        obss_generation_rate=0.01,
        npca_enabled=True,
        throughput_weight=10.0,
        latency_penalty_weight=0.1,
        history_length=10
    )
    
    # Override channels after initialization
    env.primary_channel = channels[0]
    env.npca_channel = channels[1]
    env.channels = channels
    for sta in env.stas:
        sta.primary_channel = channels[0]
        sta.npca_channel = channels[1]
    
    # Network initialization
    policy_net = DQN(n_actions=2, history_length=10).to(device)
    target_net = DQN(n_actions=2, history_length=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    steps_done = 0
    
    print(f"Starting enhanced Semi-MDP training for {num_episodes} episodes...")
    print(f"OBSS Duration: Fixed at {obss_duration} slots")
    
    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_loss = []
        decisions_made = 0
        
        while True:
            # Check if episode is already done
            if env.current_slot >= env.num_slots:
                done = True
                break
                
            # Check if we're at a decision point
            if not env._is_decision_point(env.decision_sta, env.current_slot):
                print(f"Warning: Not at decision point in episode {episode}")
                print(f"STA state: {env.decision_sta.state}, Slot: {env.current_slot}")
                # Try to advance to next decision point
                if not env._advance_to_next_decision():
                    done = True
                    break
                obs = env._get_observation()
                continue
            
            # Select action
            action = select_action(obs, policy_net, steps_done, device)
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            decisions_made += 1
            
            # Store transition
            if not done:
                memory.push(obs, action, next_obs, reward, info['duration'], done)
            else:
                memory.push(obs, action, None, reward, info['duration'], done)
            
            # Optimize model
            if len(memory) >= BATCH_SIZE:
                loss = optimize_model(policy_net, target_net, memory, optimizer, device)
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                update_target_network(policy_net, target_net)
            
            steps_done += 1
            
            if done:
                break
                
            obs = next_obs
        
        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)
        
        # Progress reporting
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            print(f"Episode {episode:3d}: Avg Reward = {avg_reward:7.3f}, "
                  f"Decisions = {decisions_made:2d}, Loss = {avg_loss:.4f}, "
                  f"Epsilon = {eps_threshold:.3f}, Memory = {len(memory)}")
    
    return episode_rewards, episode_losses, policy_net, target_net

def main():
    """Main training function"""
    print("="*60)
    print("Enhanced Semi-MDP NPCA Training")
    print("="*60)
    
    # Run training
    episode_rewards, episode_losses, policy_net, target_net = train_enhanced_semi_mdp(
        num_episodes=50,  # Reduced for quick test
        obss_duration=100
    )
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.7)
    window_size = 20
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
    plt.title('Episode Rewards (Enhanced System)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, alpha=0.7)
    plt.title('Training Loss (Enhanced System)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./enhanced_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Results summary
    print("\n" + "="*60)
    print("Enhanced Training Complete!")
    print("="*60)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Final reward (last 10): {np.mean(episode_rewards[-10:]):.3f}")
    print(f"Max reward: {max(episode_rewards):.3f}")
    print(f"Final loss: {episode_losses[-1]:.4f}")
    print(f"Memory size: {len(memory)}")
    
    # Save model
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses
    }, './enhanced_semi_mdp_model.pth')
    print("Model saved as: enhanced_semi_mdp_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()