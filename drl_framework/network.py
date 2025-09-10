import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations=None, n_actions=2, history_length=10):
        super(DQN, self).__init__()
        self.history_length = history_length
        
        # Feature extraction for basic state information
        self.basic_features = nn.Linear(7, 32)  # current_slot, backoff, cw_index, obss_remaining, 3x channel_busy
        
        # CNN for channel history processing
        self.channel_cnn = nn.Conv1d(3, 16, kernel_size=3, padding=1)  # 3 channels: primary, obss, npca
        self.channel_pool = nn.AdaptiveAvgPool1d(1)
        
        # Statistics processing
        self.stats_features = nn.Linear(2, 16)  # obss_frequency, avg_obss_duration
        
        # Combine all features
        total_features = 32 + 16 + 16  # basic + cnn + stats
        self.layer1 = nn.Linear(total_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, state_dict):
        """
        Process dictionary observation from enhanced environment
        """
        if isinstance(state_dict, dict):
            # Basic state features
            basic_state = torch.cat([
                state_dict['current_slot'].float().unsqueeze(-1) / 1000.0,  # Normalize
                state_dict['backoff_counter'].float().unsqueeze(-1) / 1024.0,
                state_dict['cw_index'].float().unsqueeze(-1) / 7.0,
                state_dict['obss_remaining'].float().unsqueeze(-1) / 100.0,
                state_dict['channel_busy_intra'].float().unsqueeze(-1),
                state_dict['channel_busy_obss'].float().unsqueeze(-1),
                state_dict['npca_channel_busy'].float().unsqueeze(-1)
            ], dim=-1)
            
            basic_features = F.relu(self.basic_features(basic_state))
            
            # Channel history processing with CNN
            channel_history = torch.stack([
                state_dict['primary_busy_history'],
                state_dict['obss_busy_history'],
                state_dict['npca_busy_history']
            ], dim=-2)  # [batch, 3, history_length]
            
            channel_features = F.relu(self.channel_cnn(channel_history))
            channel_features = self.channel_pool(channel_features).squeeze(-1)  # [batch, 16]
            
            # Statistics features
            stats = torch.cat([
                state_dict['obss_frequency'],
                state_dict['avg_obss_duration'] / 100.0  # Normalize
            ], dim=-1)
            stats_features = F.relu(self.stats_features(stats))
            
            # Combine all features
            combined = torch.cat([basic_features, channel_features, stats_features], dim=-1)
            
        else:
            # Fallback for simple tensor input (backward compatibility)
            combined = state_dict
            
        # Final layers
        x = F.relu(self.layer1(combined))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
