#!/usr/bin/env python3
"""
Train Enhanced Semi-MDP Model for NPCA Decision Making
Optimized version for faster training
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import math
import os

from npca_semi_mdp_env import NPCASemiMDPEnv
from drl_framework.network import DQN, ReplayMemory
from drl_framework.params import *

# Ensure constants are defined
TARGET_UPDATE = getattr(globals().get('TARGET_UPDATE', None), 'TARGET_UPDATE', 100)
BATCH_SIZE = getattr(globals().get('BATCH_SIZE', None), 'BATCH_SIZE', 32)
LR = getattr(globals().get('LR', None), 'LR', 1e-4)
GAMMA = getattr(globals().get('GAMMA', None), 'GAMMA', 0.99)
EPS_START = getattr(globals().get('EPS_START', None), 'EPS_START', 0.9)
EPS_END = getattr(globals().get('EPS_END', None), 'EPS_END', 0.05)
EPS_DECAY = getattr(globals().get('EPS_DECAY', None), 'EPS_DECAY', 1000)
TAU = getattr(globals().get('TAU', None), 'TAU', 0.005)

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

def select_action(obs_dict, policy_net, steps_done, device):
    """Epsilon-greedy action selection"""
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    if random.random() > eps_threshold:
        with torch.no_grad():
            state_tensor = dict_to_tensor(obs_dict, device)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()
    else:
        return random.randint(0, 1)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    """Model optimization for dictionary states"""
    if len(memory) < BATCH_SIZE:
        return None
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Process state dictionaries
    state_batch = {}
    next_state_batch = {}
    
    sample_state = batch.state[0]
    for key in sample_state.keys():
        state_batch[key] = torch.stack([dict_to_tensor(s, device)[key].squeeze(0) for s in batch.state])
        next_states = [dict_to_tensor(s, device)[key].squeeze(0) for s in batch.next_state if s is not None]
        if next_states:
            next_state_batch[key] = torch.stack(next_states)

    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    reward_batch = torch.tensor(batch.cum_reward, device=device).float()
    tau_batch = torch.tensor(batch.tau, device=device).float()
    done_batch = torch.tensor(batch.done, device=device).float()

    # Current Q values
    current_q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Next Q values
    next_q_values = torch.zeros(len(batch.state), device=device)
    non_final_mask = (done_batch == 0)
    
    if non_final_mask.sum() > 0 and next_state_batch:
        with torch.no_grad():
            next_q_values[non_final_mask] = target_net(next_state_batch).max(1)[0]

    # Semi-MDP TD target
    tau_clipped = torch.clamp(tau_batch, max=20.0)
    expected_q_values = reward_batch + (GAMMA ** tau_clipped) * next_q_values * (1.0 - done_batch)

    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()

def update_target_network(policy_net, target_net):
    """Soft target network update"""
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                     target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

def train_model(num_episodes=200, obss_duration=100, save_dir="./enhanced_results"):
    """Main training function"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Environment setup
    from drl_framework.random_access import Channel
    
    env = NPCASemiMDPEnv(
        num_stas=2, 
        num_slots=500,  # Shorter episodes for faster training
        obss_generation_rate=0.01,
        npca_enabled=True,
        throughput_weight=10.0,
        latency_penalty_weight=0.1,
        history_length=10
    )
    
    # Override with fixed OBSS duration
    channels = [
        Channel(channel_id=0, obss_generation_rate=0),
        Channel(channel_id=1, obss_generation_rate=0.01, obss_duration_range=(obss_duration, obss_duration))
    ]
    env.primary_channel = channels[0]
    env.npca_channel = channels[1] 
    env.channels = channels
    for sta in env.stas:
        sta.primary_channel = channels[0]
        sta.npca_channel = channels[1]
    
    # Network setup
    policy_net = DQN(n_actions=2, history_length=10).to(device)
    target_net = DQN(n_actions=2, history_length=10).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    steps_done = 0
    
    print(f"Starting training: {num_episodes} episodes, OBSS duration: {obss_duration}")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_loss = []
        decisions_made = 0
        
        # Episode loop
        while env.current_slot < env.num_slots:
            if not env._is_decision_point(env.decision_sta, env.current_slot):
                if not env._advance_to_next_decision():
                    break
                obs = env._get_observation()
                continue
            
            # Action selection and execution
            action = select_action(obs, policy_net, steps_done, device)
            next_obs, reward, done, _, info = env.step(action)
            
            episode_reward += reward
            decisions_made += 1
            
            # Store transition
            memory.push(obs, action, next_obs if not done else None, reward, info['duration'], done)
            
            # Model optimization
            if len(memory) >= BATCH_SIZE:
                loss = optimize_model(policy_net, target_net, memory, optimizer, device)
                if loss is not None:
                    episode_loss.append(loss)
            
            # Target network update
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
        if episode % 20 == 0 or episode == num_episodes - 1:
            recent_avg = np.mean(episode_rewards[-20:])
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            print(f"Episode {episode:3d}: Reward = {recent_avg:7.2f}, "
                  f"Decisions = {decisions_made:2d}, Loss = {avg_loss:.4f}, "
                  f"Eps = {eps_threshold:.3f}, Memory = {len(memory)}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'training_params': {
            'num_episodes': num_episodes,
            'obss_duration': obss_duration,
            'throughput_weight': 10.0,
            'latency_penalty_weight': 0.1
        }
    }, f"{save_dir}/enhanced_drl_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) >= 20:
        moving_avg = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    plt.title('Training Progress (Enhanced DRL)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_losses, alpha=0.6)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training summary
    final_avg = np.mean(episode_rewards[-20:])
    max_reward = max(episode_rewards)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Episodes trained: {num_episodes}")
    print(f"Final average reward (last 20): {final_avg:.2f}")
    print(f"Maximum reward achieved: {max_reward:.2f}")
    print(f"Total training steps: {steps_done}")
    print(f"Model saved to: {save_dir}/enhanced_drl_model.pth")
    print(f"{'='*60}")
    
    return policy_net, episode_rewards, episode_losses

def main():
    """Main execution"""
    print("="*60)
    print("Enhanced Semi-MDP NPCA Training")
    print("="*60)
    
    # Train model
    policy_net, rewards, losses = train_model(
        num_episodes=300,
        obss_duration=100,
        save_dir="./enhanced_results"
    )
    
    print("\n🎉 Enhanced model training completed successfully!")
    print("Ready for comparison with baseline methods.")

if __name__ == "__main__":
    main()