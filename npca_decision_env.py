# npca_decision_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import random
import copy

# Import from your existing NPCA simulator
from drl_framework.random_access import STA, Channel, Simulator, STAState, OccupyRequest

class DecisionMakingSTA(STA):
    """Modified STA class that pauses at decision points for RL agent input"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting_for_decision = False
        self.decision_context = None
        self.pending_action = None
        
    def requires_decision(self, slot: int) -> bool:
        """Check if this STA requires a decision at current slot"""
        return (self.state == STAState.PRIMARY_BACKOFF and 
                self.primary_channel.is_busy_by_obss(slot) and
                not self.primary_channel.is_busy_by_intra_bss(slot))
    
    def set_decision_action(self, action: int):
        """Set the action decided by RL agent"""
        self.pending_action = action
        self.waiting_for_decision = False
    
    def _handle_primary_backoff_with_decision(self, slot: int):
        """Modified primary backoff handler that waits for RL decision"""
        # 1. Primary 채널이 intra-BSS busy: frozen
        if self.primary_channel.is_busy_by_intra_bss(slot):
            self.next_state = STAState.PRIMARY_FROZEN
            return
            
        # 2. Primary 채널이 OBSS busy: Decision point!
        elif self.primary_channel.is_busy_by_obss(slot):
            if self.npca_enabled and self.npca_channel:
                # This is our decision point - wait for RL agent
                if self.pending_action is None:
                    self.waiting_for_decision = True
                    self.decision_context = {
                        'slot': slot,
                        'backoff': self.backoff,
                        'cw_index': self.cw_index,
                        'obss_remain': self.primary_channel.obss_remain,
                        'npca_channel_busy': self.npca_channel.is_busy(slot)
                    }
                    return  # Wait for decision
                
                # Execute the decided action
                if self.pending_action == 0:
                    # Action 0: Stay in PRIMARY (frozen)
                    self.next_state = STAState.PRIMARY_FROZEN
                else:
                    # Action 1: Switch to NPCA
                    self.cw_index = 0
                    self.backoff = self.generate_backoff()
                    if self.npca_channel.is_busy_by_intra_bss(slot):
                        self.next_state = STAState.NPCA_FROZEN
                    else:
                        self.next_state = STAState.NPCA_BACKOFF
                
                # Clear the action
                self.pending_action = None
            else:
                self.next_state = STAState.PRIMARY_FROZEN
        
        # 3. Primary 채널이 idle:
        else:
            if (self.backoff == 0) and not self.primary_channel.is_busy(slot):
                self.tx_remaining = self.get_tx_duration()
                self.occupy_request = OccupyRequest(
                    channel_id=self.primary_channel.channel_id, 
                    duration=self.tx_remaining, 
                    is_obss=False)
                self.next_state = STAState.PRIMARY_TX
            else:
                self.backoff -= 1 if self.backoff > 0 else 0

    def step(self, slot: int):
        """Modified step function that handles decision points"""
        if self.state == STAState.PRIMARY_BACKOFF:
            self._handle_primary_backoff_with_decision(slot)
        elif self.state == STAState.PRIMARY_FROZEN:
            self._handle_primary_frozen(slot)
        elif self.state == STAState.PRIMARY_TX:
            self._handle_primary_tx(slot)
        elif self.state == STAState.NPCA_BACKOFF:
            self._handle_npca_backoff(slot)
        elif self.state == STAState.NPCA_FROZEN:
            self._handle_npca_frozen(slot)
        elif self.state == STAState.NPCA_TX:
            self._handle_npca_tx(slot)


class DecisionMakingSimulator(Simulator):
    """Modified simulator that can pause for RL decisions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_sta_id = None  # Which STA we're making decisions for
        self.paused_for_decision = False
        self.current_slot = 0
        
    def set_decision_sta(self, sta_id: int):
        """Set which STA we'll make decisions for"""
        self.decision_sta_id = sta_id
        
    def requires_decision(self, slot: int) -> Tuple[bool, Optional[int]]:
        """Check if any STA requires a decision"""
        if self.decision_sta_id is not None:
            sta = self.stas[self.decision_sta_id]
            if isinstance(sta, DecisionMakingSTA) and sta.requires_decision(slot):
                return True, self.decision_sta_id
        return False, None
    
    def step_until_decision_or_end(self, max_steps: int = 1000) -> Tuple[bool, int, Optional[int]]:
        """Run simulation until decision point or max steps"""
        steps_taken = 0
        
        while steps_taken < max_steps and self.current_slot < self.num_slots:
            # Check if we need a decision
            needs_decision, sta_id = self.requires_decision(self.current_slot)
            if needs_decision:
                return True, steps_taken, sta_id
            
            # Run one simulation step
            self.run_single_step()
            steps_taken += 1
        
        # Reached end without decision
        return False, steps_taken, None
    
    def run_single_step(self):
        """Run a single simulation step"""
        slot = self.current_slot
        
        # ① 채널 업데이트
        for ch in self.channels:
            ch.update(slot)

        # ② STA 상태 업데이트 (decision points 처리)
        for sta in self.stas:
            sta.occupy_request = None
            sta.step(slot)

        # ③ 채널 OBSS request 수집
        for ch in self.channels:
            ch.generate_obss(slot)

        # ④ STA 전송 요청 수집
        sta_reqs = [(sta, sta.occupy_request) for sta in self.stas if sta.occupy_request is not None]

        # ⑤ 채널별로 OccupyRequest 분류 및 처리
        from collections import defaultdict
        channel_requests = defaultdict(list)
        for sta, req in sta_reqs:
            channel_requests[req.channel_id].append((sta, req))

        # ⑥ Occupy 요청 처리
        for ch_id, reqs in channel_requests.items():
            if len(reqs) == 1:
                sta, req = reqs[0]
                if req.is_obss:
                    self.channels[ch_id].add_obss_traffic(req, slot)
                else:
                    self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                if sta:
                    sta.tx_success = True
            else:
                for sta, req in reqs:
                    if sta is not None:
                        if req.is_obss:
                            self.channels[ch_id].add_obss_traffic(req, slot)
                        else:
                            self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                        sta.tx_success = False

        # ⑦ 상태 전이
        for sta in self.stas:
            sta.state = sta.next_state

        # ⑧ 로그 저장
        self.log_slot(slot)
        
        # ⑨ 다음 슬롯으로
        self.current_slot += 1


class NPCASemiMDPEnv(gym.Env):
    """
    Semi-MDP Environment for NPCA decision making
    Decisions occur when STA detects OBSS in PRIMARY_BACKOFF state
    """
    def __init__(self, 
                 num_stas: int = 2,
                 num_slots: int = 10000,
                 obss_generation_rate: float = 0.05,
                 decision_sta_id: int = 0,
                 gamma_per_slot: float = 0.99):
        super().__init__()
        
        self.num_stas = num_stas
        self.num_slots = num_slots
        self.obss_generation_rate = obss_generation_rate
        self.decision_sta_id = decision_sta_id
        self.gamma_per_slot = gamma_per_slot
        
        # Action Space: 0 = stay PRIMARY, 1 = switch to NPCA
        self.action_space = spaces.Discrete(2)
        
        # Observation Space
        self.observation_space = spaces.Dict({
            'current_slot': spaces.Discrete(num_slots),
            'backoff_counter': spaces.Discrete(1024),  # Max CW
            'cw_index': spaces.Discrete(7),  # CW stages
            'obss_remaining': spaces.Discrete(200),  # Max OBSS duration
            'primary_channel_intra_busy': spaces.Discrete(2),  # Boolean
            'primary_channel_obss_busy': spaces.Discrete(2),   # Boolean
            'npca_channel_busy': spaces.Discrete(2),   # Boolean
            'tx_success_count': spaces.Discrete(1000), # Successful transmissions
            'collision_count': spaces.Discrete(1000),  # Collisions
        })
        
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        # Initialize channels
        self.primary_channel = Channel(0, 0.0)  # No self-generated OBSS on channel 0
        self.npca_channel = Channel(1, self.obss_generation_rate)  # OBSS on channel 1
        self.channels = [self.primary_channel, self.npca_channel]
        
        # Initialize STAs with DecisionMakingSTA for decision STA
        self.stas = []
        for i in range(self.num_stas):
            if i == self.decision_sta_id:
                # Create decision-making STA
                sta = DecisionMakingSTA(
                    sta_id=i,
                    channel_id=0,  # Primary channel
                    primary_channel=self.primary_channel,
                    npca_channel=self.npca_channel,
                    npca_enabled=True
                )
            else:
                # Create regular STA
                sta = STA(
                    sta_id=i,
                    channel_id=1 if i > 0 else 0,  # STA 0 on channel 0, others on channel 1
                    primary_channel=self.channels[1 if i > 0 else 0],
                    npca_channel=self.channels[0 if i > 0 else 1],
                    npca_enabled=True
                )
            self.stas.append(sta)
        
        # Initialize simulator
        self.simulator = DecisionMakingSimulator(
            num_slots=self.num_slots,
            stas=self.stas,
            channels=self.channels
        )
        self.simulator.set_decision_sta(self.decision_sta_id)
        
        # Episode variables
        self.episode_reward = 0
        self.decision_count = 0
        
        # Advance to first decision point
        self._advance_to_next_decision()
        
        return self._get_observation(), {}
    
    def _advance_to_next_decision(self) -> bool:
        """Advance simulation to next decision point"""
        needs_decision, steps, sta_id = self.simulator.step_until_decision_or_end()
        return needs_decision
    
    def _get_observation(self) -> Dict:
        """Get current observation for decision-making STA"""
        if self.decision_sta_id >= len(self.stas):
            # Return dummy observation if STA doesn't exist
            return {
                'current_slot': 0,
                'backoff_counter': 0,
                'cw_index': 0,
                'obss_remaining': 0,
                'primary_channel_intra_busy': 0,
                'primary_channel_obss_busy': 0,
                'npca_channel_busy': 0,
                'tx_success_count': 0,
                'collision_count': 0,
            }
        
        sta = self.stas[self.decision_sta_id]
        current_slot = self.simulator.current_slot
        
        return {
            'current_slot': min(current_slot, self.num_slots - 1),
            'backoff_counter': sta.backoff,
            'cw_index': sta.cw_index,
            'obss_remaining': sta.primary_channel.obss_remain,
            'primary_channel_intra_busy': int(sta.primary_channel.is_busy_by_intra_bss(current_slot)),
            'primary_channel_obss_busy': int(sta.primary_channel.is_busy_by_obss(current_slot)),
            'npca_channel_busy': int(sta.npca_channel.is_busy(current_slot)) if sta.npca_channel else 0,
            'tx_success_count': sta.successful_transmissions if hasattr(sta, 'successful_transmissions') else 0,
            'collision_count': sta.collision_count if hasattr(sta, 'collision_count') else 0,
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action and advance to next decision point
        action: 0 = stay in PRIMARY, 1 = switch to NPCA
        """
        if self.decision_sta_id >= len(self.stas):
            # Invalid STA ID
            return self._get_observation(), 0.0, True, False, {'error': 'Invalid STA ID'}
        
        sta = self.stas[self.decision_sta_id]
        
        # Check if we're actually at a decision point
        current_slot = self.simulator.current_slot
        if not sta.requires_decision(current_slot):
            # Not at decision point - return current state
            return self._get_observation(), 0.0, False, False, {'warning': 'Not at decision point'}
        
        # Record decision context
        start_slot = current_slot
        start_successful_tx = sta.successful_transmissions if hasattr(sta, 'successful_transmissions') else 0
        start_collisions = sta.collision_count if hasattr(sta, 'collision_count') else 0
        
        # Apply action to STA
        if isinstance(sta, DecisionMakingSTA):
            sta.set_decision_action(action)
        
        # Run one more step to apply the decision
        self.simulator.run_single_step()
        
        # Simulate until next decision point or episode end
        cumulative_reward = 0.0
        duration = 1  # Already took one step above
        
        while (self.simulator.current_slot < self.num_slots and 
               duration < 500):  # Max duration limit
            
            # Check if we've reached another decision point
            if sta.requires_decision(self.simulator.current_slot):
                break
                
            # Run simulation step and calculate reward
            self.simulator.run_single_step()
            slot_reward = self._calculate_slot_reward(sta)
            cumulative_reward += slot_reward * (self.gamma_per_slot ** (duration - 1))
            duration += 1
        
        # Check if episode is done
        done = (self.simulator.current_slot >= self.num_slots)
        
        # Get final observation
        next_obs = self._get_observation() if not done else self._get_observation()
        
        # Calculate final metrics
        end_successful_tx = sta.successful_transmissions if hasattr(sta, 'successful_transmissions') else 0
        end_collisions = sta.collision_count if hasattr(sta, 'collision_count') else 0
        
        info = {
            'duration': duration,
            'start_slot': start_slot,
            'end_slot': self.simulator.current_slot,
            'decision_count': self.decision_count,
            'action_taken': action,
            'successful_tx_gained': end_successful_tx - start_successful_tx,
            'collisions_incurred': end_collisions - start_collisions,
        }
        
        self.decision_count += 1
        
        return next_obs, cumulative_reward, done, False, info
    
    def _calculate_slot_reward(self, sta) -> float:
        """Calculate reward for current slot"""
        reward = 0.0
        
        # Reward for successful transmission
        if hasattr(sta, 'tx_success') and getattr(sta, 'tx_success', False):
            if sta.state == STAState.PRIMARY_TX:
                reward += 10.0  # High reward for primary transmission success
            elif sta.state == STAState.NPCA_TX:
                reward += 8.0   # Slightly lower for NPCA transmission success
        
        # Small penalty for waiting states (encourages efficiency)
        if sta.state in [STAState.PRIMARY_FROZEN, STAState.NPCA_FROZEN]:
            reward -= 0.1
        
        # Small penalty for backoff states
        if sta.state in [STAState.PRIMARY_BACKOFF, STAState.NPCA_BACKOFF]:
            reward -= 0.05
        
        # Penalty for collision
        if hasattr(sta, 'tx_success') and getattr(sta, 'tx_success', True) == False:
            reward -= 5.0
        
        return reward
    
    def render(self, mode='human'):
        """Render current state"""
        if mode == 'human':
            sta = self.stas[self.decision_sta_id] if self.decision_sta_id < len(self.stas) else None
            if sta:
                print(f"Slot: {self.simulator.current_slot}, STA {self.decision_sta_id} State: {sta.state.name}, "
                      f"Backoff: {sta.backoff}, Decisions: {self.decision_count}")
            else:
                print(f"Slot: {self.simulator.current_slot}, Invalid STA ID: {self.decision_sta_id}")


# Test the environment
def test_npca_decision_env():
    """Test the NPCA decision-making environment"""
    print("🧪 Testing NPCA Decision-Making Environment...")
    
    env = NPCASemiMDPEnv(
        num_stas=3,
        num_slots=5000,
        obss_generation_rate=0.02,
        decision_sta_id=0
    )
    
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    
    episode_reward = 0
    step_count = 0
    
    for _ in range(50):  # Test up to 50 decisions
        # Random action for testing
        action = random.choice([0, 1])
        action_name = "Stay PRIMARY" if action == 0 else "Switch to NPCA"
        
        print(f"\nStep {step_count + 1}: Taking action {action} ({action_name})")
        
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        print(f"  Reward: {reward:.3f}")
        print(f"  Duration: {info['duration']} slots")
        print(f"  TX Success: {info['successful_tx_gained']}, Collisions: {info['collisions_incurred']}")
        print(f"  Next observation: {obs}")
        
        if done:
            print("Episode finished!")
            break
    
    print(f"\nEpisode Summary:")
    print(f"  Total reward: {episode_reward:.3f}")
    print(f"  Total decisions: {step_count}")
    print(f"  Final slot: {env.simulator.current_slot}")

if __name__ == "__main__":
    test_npca_decision_env()