# npca_semi_mdp_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import random

# Import from your random_access.py
from drl_framework.random_access import STA, Channel, Simulator, STAState, OccupyRequest

class NPCASemiMDPEnv(gym.Env):
    """
    Semi-MDP Environment for NPCA decision making
    Decision points occur when STA is in PRIMARY_BACKOFF and detects OBSS
    """
    def __init__(self, 
                 num_stas: int = 2,
                 num_slots: int = 1000,
                 obss_generation_rate: float = 0.1,
                 npca_enabled: bool = True):
        super().__init__()
        
        self.num_stas = num_stas
        self.num_slots = num_slots
        self.obss_generation_rate = obss_generation_rate
        self.npca_enabled = npca_enabled
        
        # Action Space: 0 = stay in PRIMARY (frozen), 1 = switch to NPCA
        self.action_space = spaces.Discrete(2)
        
        # Observation Space - 각 STA와 채널의 상태 정보
        self.observation_space = spaces.Dict({
            'current_slot': spaces.Discrete(num_slots),
            'backoff_counter': spaces.Discrete(1024),  # Max CW
            'cw_index': spaces.Discrete(7),  # CW stages
            'obss_remaining': spaces.Discrete(100),  # Max OBSS duration
            'channel_busy_intra': spaces.Discrete(2),  # Boolean
            'channel_busy_obss': spaces.Discrete(2),   # Boolean
            'npca_channel_busy': spaces.Discrete(2),   # Boolean
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Initialize channels
        self.primary_channel = Channel(0, self.obss_generation_rate)
        self.npca_channel = Channel(1, 0.0) if self.npca_enabled else None
        self.channels = [self.primary_channel, self.npca_channel] if self.npca_channel else [self.primary_channel]
        
        # Initialize STAs
        self.stas = []
        for i in range(self.num_stas):
            sta = STA(
                sta_id=i,
                channel_id=0,  # Primary channel
                primary_channel=self.primary_channel,
                npca_channel=self.npca_channel,
                npca_enabled=self.npca_enabled
            )
            self.stas.append(sta)
        
        # Current decision-making STA (we'll focus on STA 0 for simplicity)
        self.decision_sta = self.stas[0]
        
        # Simulator state
        self.current_slot = 0
        self.episode_reward = 0
        self.decision_count = 0
        
        # Find first decision point
        self._advance_to_next_decision()
        
        return self._get_observation(), {}
    
    def _is_decision_point(self, sta: STA, slot: int) -> bool:
        """
        Check if current state requires decision making
        Decision needed when: PRIMARY_BACKOFF + OBSS detected
        (regardless of backoff counter value)
        """
        return (sta.state == STAState.PRIMARY_BACKOFF and 
                sta.primary_channel.is_busy_by_obss(slot))
    
    def _advance_to_next_decision(self) -> bool:
        """
        Advance simulation until next decision point or episode end
        Returns True if decision point found, False if episode ended
        """
        max_advance = 1000  # Prevent infinite loops
        advance_count = 0
        
        while (self.current_slot < self.num_slots and 
               advance_count < max_advance and
               not self._is_decision_point(self.decision_sta, self.current_slot)):
            
            # Update channels
            for ch in self.channels:
                ch.update(self.current_slot)
                ch.generate_obss(self.current_slot)
            
            # Update all STAs (passive simulation)
            for sta in self.stas:
                sta.occupy_request = None
                sta.step(self.current_slot)
                sta.state = sta.next_state
            
            self.current_slot += 1
            advance_count += 1
        
        return self.current_slot < self.num_slots and advance_count < max_advance
    
    def _get_observation(self) -> Dict:
        """Get current observation for the decision-making STA"""
        sta = self.decision_sta
        
        return {
            'current_slot': self.current_slot,
            'backoff_counter': sta.backoff,
            'cw_index': sta.cw_index,
            'obss_remaining': sta.primary_channel.obss_remain,
            'channel_busy_intra': int(sta.primary_channel.is_busy_by_intra_bss(self.current_slot)),
            'channel_busy_obss': int(sta.primary_channel.is_busy_by_obss(self.current_slot)),
            'npca_channel_busy': int(sta.npca_channel.is_busy(self.current_slot)) if sta.npca_channel else 0,
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action and advance to next decision point
        action: 0 = stay in PRIMARY (frozen), 1 = switch to NPCA
        """
        if not self._is_decision_point(self.decision_sta, self.current_slot):
            raise ValueError("step() called when not at decision point!")
        
        sta = self.decision_sta
        start_slot = self.current_slot
        
        # Apply action - decision between staying primary vs switching to NPCA
        if action == 0:
            # Stay in primary channel -> PRIMARY_FROZEN (wait for OBSS to end)
            sta.next_state = STAState.PRIMARY_FROZEN
        else:
            # Switch to NPCA channel
            if sta.npca_enabled and sta.npca_channel:
                # Reset CW and backoff for NPCA
                sta.cw_index = 0
                sta.backoff = sta.generate_backoff()
                
                # Check NPCA channel status
                if sta.npca_channel.is_busy_by_intra_bss(self.current_slot):
                    sta.next_state = STAState.NPCA_FROZEN
                else:
                    sta.next_state = STAState.NPCA_BACKOFF
            else:
                # Fallback to primary frozen if NPCA not available
                sta.next_state = STAState.PRIMARY_FROZEN
        
        sta.state = sta.next_state
        
        # Simulate until next decision point - 지연된 보상으로 수정
        cumulative_reward = 0  # 슬롯별 보상 대신 옵션 종료 시 보상 계산
        duration = 0
        
        # 옵션 시작 시점의 channel_occupancy_time 기록
        initial_occupancy = sta.channel_occupancy_time
        
        while (self.current_slot < self.num_slots and 
               duration < 500 and  # Max duration limit
               not self._is_decision_point(sta, self.current_slot)):
            
            # Update channels
            for ch in self.channels:
                ch.update(self.current_slot)
                ch.generate_obss(self.current_slot)
            
            # Update all STAs
            for s in self.stas:
                s.occupy_request = None
                s.step(self.current_slot)
                s.state = s.next_state
            
            self.current_slot += 1
            duration += 1
        
        # 옵션 종료 시 보상 계산: 옵션 기간 동안 성공적으로 전송한 슬롯 수
        option_successful_slots = sta.channel_occupancy_time - initial_occupancy
        cumulative_reward = float(option_successful_slots)  # 성공 전송 슬롯 수를 보상으로 사용
        
        # Check if episode is done
        done = (self.current_slot >= self.num_slots)
        
        # Get next observation
        next_obs = self._get_observation() if not done else {}
        
        info = {
            'duration': duration,
            'start_slot': start_slot,
            'end_slot': self.current_slot,
            'decision_count': self.decision_count
        }
        
        self.decision_count += 1
        
        return next_obs, cumulative_reward, done, False, info
    
    def _calculate_slot_reward(self, sta: STA) -> float:
        """지연된 보상 구조: 슬롯별 즉시 보상 제거"""
        # 슬롯별 즉시 보상 제거 - 모든 보상은 에피소드 종료 시에만 계산
        # 실제 보상은 에피소드 종료 시 성공적으로 전송한 슬롯 수로 계산됨
        return 0.0
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        if mode == 'human':
            print(f"Slot: {self.current_slot}, STA State: {self.decision_sta.state.name}, "
                  f"Backoff: {self.decision_sta.backoff}, Decisions: {self.decision_count}")


# Test the environment
if __name__ == "__main__":
    env = NPCASemiMDPEnv(num_stas=2, num_slots=1000)
    
    print("Testing Semi-MDP NPCA Environment...")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Decision point - Primary backoff with OBSS detected!")
    
    for step_count in range(10):  # Test 10 decisions
        # Random action for testing
        action = random.choice([0, 1])
        action_name = "Stay PRIMARY" if action == 0 else "Switch to NPCA"
        print(f"\nStep {step_count + 1}: Taking action {action} ({action_name})")
        
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.3f}")
        print(f"Duration: {info['duration']} slots")
        print(f"Next observation: {obs}")
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nTotal decisions made: {env.decision_count}")