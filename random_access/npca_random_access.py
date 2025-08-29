import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from random_access.random_access import STAFiniteStateMachine, STAState, FrameInfo, CONTENTION_WINDOW

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from random_access.random_access import STAFiniteStateMachine, STAState, FrameInfo, CONTENTION_WINDOW

class CrossChannelNPCASTAFiniteStateMachine(STAFiniteStateMachine):
    """Cross-Channel NPCA Station implementation
    
    Key Feature:
    - When OBSS occurs on primary channel, NPCA switches to alternative channel
    - NPCA transmission on alternative channel becomes OBSS for that channel
    """
    
    def __init__(self, sta_id: int, channel_id: int, npca_enabled: bool = True, 
                 alternative_channel: int = 0):
        super().__init__(sta_id, channel_id)
        self.npca_enabled = npca_enabled
        self.primary_channel = channel_id
        self.alternative_channel = alternative_channel
        self.current_active_channel = channel_id  # Currently transmitting on which channel
        
        # NPCA statistics
        self.obss_immunity_activations = 0
        self.channel_switches = 0  # Times switched to alternative channel
        self.alternative_channel_transmissions = 0  # Successful transmissions on alt channel
        
        # Channel state trackers (will be set by simulation)
        self.primary_channel_tracker = None
        self.alternative_channel_tracker = None
        
    def set_channel_trackers(self, primary_tracker, alternative_tracker):
        """Set references to channel FSMs for cross-channel operation"""
        self.primary_channel_tracker = primary_tracker
        self.alternative_channel_tracker = alternative_tracker
    
    def update(self, current_slot: int, channel_busy: bool, obss_busy: bool = False) -> Tuple[bool, int]:
        """Cross-Channel NPCA FSM update
        
        Returns:
            Tuple[bool, int]: (tx_attempt, target_channel)
        """
        
        self.tx_attempt = False
        target_channel = self.primary_channel  # Default to primary channel
        
        # Skip update if currently transmitting and transmission not finished
        if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
            return False, self.current_active_channel
        
        # Standard deferral counting (only for primary channel activity)
        if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
            if channel_busy and not obss_busy:
                self.intra_bss_deferrals += 1
            elif obss_busy and not channel_busy:
                if not self.npca_enabled:
                    self.obss_deferrals += 1
            elif channel_busy and obss_busy:
                self.intra_bss_deferrals += 1
        
        # Main FSM logic with Cross-Channel NPCA enhancement
        if self.state == STAState.IDLE:
            self._handle_idle_state(current_slot)
            
        elif self.state == STAState.BACKOFF:
            tx_attempt, target_channel = self._cross_channel_handle_backoff(current_slot, channel_busy, obss_busy)
            self.tx_attempt = tx_attempt
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._cross_channel_handle_backoff_frozen(current_slot, channel_busy, obss_busy)
            
        elif self.state == STAState.OBSS_FROZEN:
            self._cross_channel_handle_obss_frozen(current_slot, channel_busy, obss_busy)
            
        elif self.state == STAState.TRANSMITTING:
            self._handle_transmitting(current_slot)
            target_channel = self.current_active_channel
        
        return self.tx_attempt, target_channel
    
    def _cross_channel_handle_backoff(self, current_slot: int, channel_busy: bool, obss_busy: bool) -> Tuple[bool, int]:
        """Cross-channel NPCA backoff handling"""
        target_channel = self.primary_channel
        
        if channel_busy:
            # Always defer to intra-BSS traffic on primary channel
            self.state = STAState.BACKOFF_FROZEN
            return False, target_channel
            
        elif obss_busy and not self.npca_enabled:
            # Legacy behavior: freeze on OBSS
            self.state = STAState.OBSS_FROZEN
            return False, target_channel
            
        elif obss_busy and self.npca_enabled:
            # NPCA behavior: try to switch to alternative channel
            if self._can_use_alternative_channel(current_slot):
                self.obss_immunity_activations += 1
                self.channel_switches += 1
                target_channel = self.alternative_channel
                
                if self.backoff_counter == 0:
                    self.state = STAState.TRANSMITTING
                    self.current_active_channel = target_channel
                    self.tx_attempt = True
                    self.total_attempts += 1
                    return True, target_channel
                else:
                    self.backoff_counter -= 1
                    if self.backoff_counter == 0:
                        self.state = STAState.TRANSMITTING
                        self.current_active_channel = target_channel
                        self.tx_attempt = True
                        self.total_attempts += 1
                        return True, target_channel
            else:
                # Can't use alternative channel - wait
                self.state = STAState.BACKOFF_FROZEN
                return False, target_channel
        else:
            # Channel clear on primary - proceed normally
            if self.backoff_counter == 0:
                self.state = STAState.TRANSMITTING
                self.current_active_channel = self.primary_channel
                self.tx_attempt = True
                self.total_attempts += 1
                return True, target_channel
            else:
                self.backoff_counter -= 1
                if self.backoff_counter == 0:
                    self.state = STAState.TRANSMITTING
                    self.current_active_channel = self.primary_channel
                    self.tx_attempt = True
                    self.total_attempts += 1
                    return True, target_channel
        
        return False, target_channel
    
    def _cross_channel_handle_backoff_frozen(self, current_slot: int, channel_busy: bool, obss_busy: bool):
        """Cross-channel NPCA frozen state handling"""
        if not channel_busy and not obss_busy:
            # Both cleared - resume backoff on primary
            self.state = STAState.BACKOFF
        elif not channel_busy and obss_busy:
            if self.npca_enabled and self._can_use_alternative_channel(current_slot):
                # NPCA: Try alternative channel despite OBSS on primary
                self.state = STAState.BACKOFF
                self.obss_immunity_activations += 1
            else:
                # Legacy or can't use alternative channel
                if not self.npca_enabled:
                    self.state = STAState.OBSS_FROZEN
    
    def _cross_channel_handle_obss_frozen(self, current_slot: int, channel_busy: bool, obss_busy: bool):
        """Handle OBSS frozen state (mainly for legacy mode)"""
        if channel_busy:
            self.state = STAState.BACKOFF_FROZEN
        elif not obss_busy:
            self.state = STAState.BACKOFF
    
    def _can_use_alternative_channel(self, current_slot: int) -> bool:
        """Check if alternative channel is available for NPCA transmission"""
        if not self.alternative_channel_tracker:
            return True  # No tracker info - assume available
        
        # Check if alternative channel is busy with intra-BSS traffic
        alternative_busy = self.alternative_channel_tracker.is_busy(current_slot)
        return not alternative_busy  # Can use if not busy with intra-BSS
    
    def on_transmission_result(self, result: str, completion_slot: int):
        """Handle transmission result with channel tracking"""
        if self.state != STAState.TRANSMITTING:
            return
            
        if result == 'success':
            self.state = STAState.IDLE
            self.successful_transmissions += 1
            self.last_successful_tx_slot = completion_slot
            
            # Track alternative channel transmissions
            if self.current_active_channel == self.alternative_channel:
                self.alternative_channel_transmissions += 1
            
            self._reset_transmission_params()
            
        elif result == 'collision':
            # Standard collision handling
            self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
            self.backoff_counter = self.get_new_backoff()
            self.collision_count += 1
            self.state = STAState.BACKOFF
            self._reset_transmission_params(keep_frame=True)
        
        # Reset active channel to primary after transmission
        self.current_active_channel = self.primary_channel
    
    def get_npca_statistics(self) -> Dict:
        """Get Cross-Channel NPCA statistics"""
        return {
            'npca_enabled': self.npca_enabled,
            'primary_channel': self.primary_channel,
            'alternative_channel': self.alternative_channel,
            'obss_immunity_activations': self.obss_immunity_activations,
            'channel_switches': self.channel_switches,
            'alternative_channel_transmissions': self.alternative_channel_transmissions,
            'alternative_channel_success_rate': (
                self.alternative_channel_transmissions / max(1, self.channel_switches)
            ),
            'cross_channel_npca': True
        }


@dataclass
class NPCAOBSSTraffic:
    """NPCA-generated OBSS traffic information"""
    npca_sta_id: int
    start_slot: int
    duration: int
    source_channel: int  # NPCA's primary channel
    target_channel: int  # Channel where NPCA transmits (becomes OBSS)
    npca_generated: bool = True


class CrossChannelNPCASimulation:
    """Simulation modifications for Cross-Channel NPCA
    
    Key changes:
    1. Channel 0: No self-generated OBSS, only receives NPCA OBSS from Channel 1
    2. Channel 1: Self-generated OBSS + NPCA STAs that can switch to Channel 0
    3. NPCA transmissions on Channel 0 become OBSS for Channel 0 Legacy STAs
    """
    
    def __init__(self, base_simulation):
        self.base_sim = base_simulation
        self.npca_obss_traffic = []  # Track NPCA-generated OBSS
    
    def modify_obss_generation(self):
        """Modify OBSS generation to be channel-specific"""
        # Remove OBSS generator for channel 0 (no self-generated OBSS)
        if len(self.base_sim.obss_generators) > 0:
            # Keep only channel 1 OBSS generator
            self.base_sim.obss_generators = [self.base_sim.obss_generators[1]] if len(self.base_sim.obss_generators) > 1 else []
            
            # Update source channel to ensure it's channel 1
            if self.base_sim.obss_generators:
                self.base_sim.obss_generators[0].source_channel = 1
    
    def handle_npca_transmission(self, npca_sta, current_slot: int, target_channel: int):
        """Handle NPCA transmission that creates OBSS on target channel"""
        if target_channel != npca_sta.primary_channel and npca_sta.current_frame:
            # NPCA is transmitting on alternative channel - create OBSS
            npca_obss = NPCAOBSSTraffic(
                npca_sta_id=npca_sta.sta_id,
                start_slot=current_slot,
                duration=npca_sta.current_frame.size,
                source_channel=npca_sta.primary_channel,
                target_channel=target_channel,
                npca_generated=True
            )
            
            # Add to target channel as OBSS
            if 0 <= target_channel < len(self.base_sim.channels):
                from random_access.random_access import OBSSTraffic
                obss_traffic = OBSSTraffic(
                    obss_id=f"npca_{npca_sta.sta_id}_{current_slot}",
                    start_slot=current_slot,
                    duration=npca_sta.current_frame.size,
                    source_channel=npca_sta.primary_channel
                )
                self.base_sim.channels[target_channel].add_obss_traffic(obss_traffic)
                
            self.npca_obss_traffic.append(npca_obss)
    
    def get_npca_obss_statistics(self) -> Dict:
        """Get statistics about NPCA-generated OBSS traffic"""
        channel_0_npca_obss = [t for t in self.npca_obss_traffic if t.target_channel == 0]
        
        return {
            'total_npca_obss_events': len(self.npca_obss_traffic),
            'channel_0_npca_obss_events': len(channel_0_npca_obss),
            'total_npca_obss_duration': sum(t.duration for t in self.npca_obss_traffic),
            'channel_0_npca_obss_duration': sum(t.duration for t in channel_0_npca_obss),
        }


# class DynamicNPCASTAFiniteStateMachine(STAFiniteStateMachine):
#     """Dynamic NPCA with Frame Duration Adaptation
    
#     Key Features:
#     1. OBSS Immunity: Ignores OBSS traffic during backoff
#     2. Dynamic Frame Duration: Adjusts frame size to not exceed primary BSS occupied time
#     3. Primary BSS Awareness: Respects intra-BSS transmissions
#     """
    
#     def __init__(self, sta_id: int, channel_id: int, npca_enabled: bool = True, 
#                  max_frame_size: int = 33, min_frame_size: int = 10):
#         super().__init__(sta_id, channel_id)
#         self.npca_enabled = npca_enabled
        
#         # Dynamic frame duration parameters
#         self.max_frame_size = max_frame_size  # Original frame size
#         self.min_frame_size = min_frame_size  # Minimum allowed frame size
#         self.adaptive_frame_size = max_frame_size  # Current adaptive frame size
        
#         # NPCA statistics
#         self.obss_immunity_activations = 0
#         self.frame_adaptations = 0  # Times frame size was adapted
#         self.truncated_frames = 0   # Times frame was truncated
        
#         # Channel awareness
#         self.channel_state_tracker = None  # Will be set by simulation
        
#     def set_channel_tracker(self, channel_fsm):
#         """Set reference to channel FSM for dynamic duration calculation"""
#         self.channel_state_tracker = channel_fsm
    
#     def calculate_adaptive_frame_size(self, current_slot: int) -> int:
#         """Calculate adaptive frame size based on primary BSS occupancy"""
#         if not self.npca_enabled or not self.channel_state_tracker:
#             return self.max_frame_size
        
#         # Get primary BSS occupied until time
#         primary_bss_end = self.channel_state_tracker.occupied_until
        
#         if primary_bss_end <= current_slot:
#             # No primary BSS activity - use full frame size
#             return self.max_frame_size
        
#         # Calculate available slots until primary BSS ends
#         available_slots = max(0, primary_bss_end - current_slot)
        
#         if available_slots == 0:
#             # No space available - use minimum frame
#             adapted_size = self.min_frame_size
#         elif available_slots >= self.max_frame_size:
#             # Enough space for full frame
#             adapted_size = self.max_frame_size
#         else:
#             # Partial space - adapt frame size
#             adapted_size = max(self.min_frame_size, available_slots)
#             self.frame_adaptations += 1
        
#         # Track if frame was truncated
#         if adapted_size < self.max_frame_size:
#             self.truncated_frames += 1
        
#         return adapted_size
    
#     def update(self, current_slot: int, channel_busy: bool, obss_busy: bool = False) -> bool:
#         """Dynamic NPCA FSM update with frame duration adaptation"""
        
#         self.tx_attempt = False
        
#         # Skip update if currently transmitting and transmission not finished
#         if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
#             return False
        
#         # Standard deferral counting
#         if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
#             if channel_busy and not obss_busy:
#                 self.intra_bss_deferrals += 1
#             elif obss_busy and not channel_busy:
#                 if not self.npca_enabled:
#                     self.obss_deferrals += 1
#             elif channel_busy and obss_busy:
#                 self.intra_bss_deferrals += 1
        
#         # Main FSM logic with dynamic NPCA enhancement
#         if self.state == STAState.IDLE:
#             self._handle_idle_state(current_slot)
            
#         elif self.state == STAState.BACKOFF:
#             self._dynamic_handle_backoff(current_slot, channel_busy, obss_busy)
            
#         elif self.state == STAState.BACKOFF_FROZEN:
#             self._dynamic_handle_backoff_frozen(current_slot, channel_busy, obss_busy)
            
#         elif self.state == STAState.OBSS_FROZEN:
#             self._dynamic_handle_obss_frozen(current_slot, channel_busy, obss_busy)
            
#         elif self.state == STAState.TRANSMITTING:
#             self._handle_transmitting(current_slot)
        
#         return self.tx_attempt
    
#     def _dynamic_handle_backoff(self, current_slot: int, channel_busy: bool, obss_busy: bool):
#         """Dynamic NPCA backoff handling with frame adaptation"""
#         if channel_busy:
#             # Always defer to intra-BSS traffic
#             self.state = STAState.BACKOFF_FROZEN
#         elif obss_busy and not self.npca_enabled:
#             # Legacy behavior: freeze on OBSS
#             self.state = STAState.OBSS_FROZEN
#         elif obss_busy and self.npca_enabled:
#             # NPCA behavior: ignore OBSS, but check if transmission is viable
#             if self._can_transmit_without_interfering(current_slot):
#                 self.obss_immunity_activations += 1
#                 self._proceed_with_adaptive_backoff(current_slot)
#             else:
#                 # Can't transmit without interfering with primary BSS - wait
#                 self.state = STAState.BACKOFF_FROZEN
#         else:
#             # Channel clear - proceed with backoff
#             self._proceed_with_adaptive_backoff(current_slot)
    
#     def _dynamic_handle_backoff_frozen(self, current_slot: int, channel_busy: bool, obss_busy: bool):
#         """Dynamic NPCA frozen state handling"""
#         if not channel_busy and not obss_busy:
#             # Both cleared - resume backoff
#             self.state = STAState.BACKOFF
#         elif not channel_busy and obss_busy:
#             if self.npca_enabled and self._can_transmit_without_interfering(current_slot):
#                 # NPCA: Resume backoff despite OBSS if no primary BSS interference
#                 self.state = STAState.BACKOFF
#                 self.obss_immunity_activations += 1
#             else:
#                 # Legacy or can't transmit without interfering
#                 if not self.npca_enabled:
#                     self.state = STAState.OBSS_FROZEN
    
#     def _dynamic_handle_obss_frozen(self, current_slot: int, channel_busy: bool, obss_busy: bool):
#         """Handle OBSS frozen state (mainly for legacy mode)"""
#         if channel_busy:
#             self.state = STAState.BACKOFF_FROZEN
#         elif not obss_busy:
#             self.state = STAState.BACKOFF
    
#     def _can_transmit_without_interfering(self, current_slot: int) -> bool:
#         """Check if NPCA can transmit without interfering with primary BSS"""
#         if not self.channel_state_tracker:
#             return True  # No channel info - assume safe
        
#         # Calculate minimum viable frame size
#         min_viable_size = self.calculate_adaptive_frame_size(current_slot)
        
#         # Can transmit if we can send at least minimum frame size
#         return min_viable_size >= self.min_frame_size
    
#     def _proceed_with_adaptive_backoff(self, current_slot: int):
#         """Proceed with backoff and prepare adaptive frame if needed"""
#         if self.backoff_counter == 0:
#             # Calculate adaptive frame size for transmission
#             self.adaptive_frame_size = self.calculate_adaptive_frame_size(current_slot)
            
#             # Update current frame size if it exists
#             if self.current_frame:
#                 self.current_frame.size = self.adaptive_frame_size
            
#             self.state = STAState.TRANSMITTING
#             self.tx_attempt = True
#             self.total_attempts += 1
#         else:
#             self.backoff_counter -= 1
#             if self.backoff_counter == 0:
#                 # Calculate adaptive frame size for transmission
#                 self.adaptive_frame_size = self.calculate_adaptive_frame_size(current_slot)
                
#                 # Update current frame size if it exists
#                 if self.current_frame:
#                     self.current_frame.size = self.adaptive_frame_size
                
#                 self.state = STAState.TRANSMITTING
#                 self.tx_attempt = True
#                 self.total_attempts += 1
    
#     def _handle_idle_state(self, current_slot: int):
#         """Handle IDLE state with frame preparation"""
#         if self.has_frame_to_send and self.tx_queue:
#             self.current_frame = self.tx_queue.pop(0)
#             self.has_frame_to_send = len(self.tx_queue) > 0
            
#             # Set initial frame size (will be adapted during transmission)
#             self.current_frame.size = self.max_frame_size
#             self.frame_creation_slot = self.current_frame.creation_slot
            
#             # Start with random backoff
#             self.backoff_counter = self.get_new_backoff()
#             self.state = STAState.BACKOFF
    
#     def get_npca_statistics(self) -> Dict:
#         """Get Dynamic NPCA statistics"""
#         return {
#             'npca_enabled': self.npca_enabled,
#             'obss_immunity_activations': self.obss_immunity_activations,
#             'frame_adaptations': self.frame_adaptations,
#             'truncated_frames': self.truncated_frames,
#             'max_frame_size': self.max_frame_size,
#             'min_frame_size': self.min_frame_size,
#             'current_adaptive_size': self.adaptive_frame_size,
#             'dynamic_npca': True
#         }
    
#     def get_frame_efficiency(self) -> float:
#         """Calculate frame size efficiency (adapted vs original)"""
#         if self.total_attempts == 0:
#             return 1.0
#         return 1.0 - (self.truncated_frames / self.total_attempts)


# class DynamicFrameInfo(FrameInfo):
#     """Extended Frame Info with dynamic size capability"""
    
#     def __init__(self, frame_id: int, source: int, size: int, timestamp: int, 
#                  creation_slot: int, original_size: int = None):
#         super().__init__(frame_id, source, size, timestamp, creation_slot)
#         self.original_size = original_size or size
#         self.adapted = False
    
#     def adapt_size(self, new_size: int):
#         """Adapt frame size dynamically"""
#         if new_size != self.size:
#             self.adapted = True
#             self.size = new_size

# class VanillaNPCASTAFiniteStateMachine(STAFiniteStateMachine):
#     """Vanilla NPCA (Next-generation Primary Channel Access) Station implementation
    
#     Core NPCA Feature:
#     - OBSS Immunity: Does not freeze backoff during OBSS traffic (that's it!)
    
#     No additional optimizations, no adaptive backoff, no smart collision detection.
#     Pure NPCA behavior for clean evaluation.
#     """
    
#     def __init__(self, sta_id: int, channel_id: int, npca_enabled: bool = True):
#         super().__init__(sta_id, channel_id)
#         self.npca_enabled = npca_enabled
        
#         # Simple NPCA statistics (minimal)
#         self.obss_immunity_activations = 0  # Times NPCA ignored OBSS
        
#     def update(self, current_slot: int, channel_busy: bool, obss_busy: bool = False) -> bool:
#         """Vanilla NPCA FSM update with simple OBSS immunity"""
        
#         self.tx_attempt = False
        
#         # Skip update if currently transmitting and transmission not finished
#         if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
#             return False
        
#         # Standard deferral counting (same as legacy)
#         if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
#             if channel_busy and not obss_busy:
#                 self.intra_bss_deferrals += 1
#             elif obss_busy and not channel_busy:
#                 # Only count OBSS deferrals if we actually defer (NPCA might not)
#                 if not self.npca_enabled:
#                     self.obss_deferrals += 1
#             elif channel_busy and obss_busy:
#                 self.intra_bss_deferrals += 1  # Intra-BSS takes priority
        
#         # Main FSM logic with vanilla NPCA enhancement
#         if self.state == STAState.IDLE:
#             self._handle_idle_state(current_slot)
            
#         elif self.state == STAState.BACKOFF:
#             self._vanilla_handle_backoff(channel_busy, obss_busy)
            
#         elif self.state == STAState.BACKOFF_FROZEN:
#             self._vanilla_handle_backoff_frozen(channel_busy, obss_busy)
            
#         elif self.state == STAState.OBSS_FROZEN:
#             self._vanilla_handle_obss_frozen(channel_busy, obss_busy)
            
#         elif self.state == STAState.TRANSMITTING:
#             self._handle_transmitting(current_slot)
        
#         return self.tx_attempt
    
#     def _vanilla_handle_backoff(self, channel_busy: bool, obss_busy: bool):
#         """Vanilla NPCA backoff handling - only OBSS immunity"""
#         if channel_busy:
#             # Always defer to intra-BSS traffic (same as legacy)
#             self.state = STAState.BACKOFF_FROZEN
#         elif obss_busy and not self.npca_enabled:
#             # Legacy behavior: freeze on OBSS
#             self.state = STAState.OBSS_FROZEN
#         elif obss_busy and self.npca_enabled:
#             # NPCA behavior: ignore OBSS, continue backoff
#             self.obss_immunity_activations += 1
#             self._proceed_with_backoff()
#         else:
#             # Channel clear - proceed with backoff
#             self._proceed_with_backoff()
    
#     def _vanilla_handle_backoff_frozen(self, channel_busy: bool, obss_busy: bool):
#         """Vanilla NPCA frozen state handling"""
#         if not channel_busy and not obss_busy:
#             # Both cleared - resume backoff
#             self.state = STAState.BACKOFF
#         elif not channel_busy and obss_busy:
#             if self.npca_enabled:
#                 # NPCA: Resume backoff despite OBSS
#                 self.state = STAState.BACKOFF
#                 self.obss_immunity_activations += 1
#             else:
#                 # Legacy: Transition to OBSS_FROZEN
#                 self.state = STAState.OBSS_FROZEN
    
#     def _vanilla_handle_obss_frozen(self, channel_busy: bool, obss_busy: bool):
#         """Handle OBSS frozen state (mainly for legacy mode)"""
#         if channel_busy:
#             # Intra-BSS appeared - higher priority
#             self.state = STAState.BACKOFF_FROZEN
#         elif not obss_busy:
#             # OBSS cleared - resume backoff
#             self.state = STAState.BACKOFF
    
#     def _proceed_with_backoff(self):
#         """Standard backoff countdown (no NPCA modifications)"""
#         if self.backoff_counter == 0:
#             self.state = STAState.TRANSMITTING
#             self.tx_attempt = True
#             self.total_attempts += 1
#         else:
#             self.backoff_counter -= 1
#             if self.backoff_counter == 0:
#                 self.state = STAState.TRANSMITTING
#                 self.tx_attempt = True
#                 self.total_attempts += 1
    
#     def on_transmission_result(self, result: str, completion_slot: int):
#         """Standard transmission result handling (no NPCA modifications)"""
#         if self.state != STAState.TRANSMITTING:
#             return
            
#         if result == 'success':
#             self.state = STAState.IDLE
#             self.successful_transmissions += 1
#             self.last_successful_tx_slot = completion_slot
#             self._reset_transmission_params()
            
#         elif result == 'collision':
#             # Standard collision handling (no NPCA modifications)
#             self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
#             self.backoff_counter = self.get_new_backoff()
#             self.collision_count += 1
#             self.state = STAState.BACKOFF
#             self._reset_transmission_params(keep_frame=True)
    
#     def get_new_backoff(self) -> int:
#         """Standard backoff generation (no NPCA modifications)"""
#         cw_index = min(self.backoff_stage, len(CONTENTION_WINDOW) - 1)
#         cw = CONTENTION_WINDOW[cw_index]
#         return np.random.randint(0, cw + 1)
    
#     def get_npca_statistics(self) -> Dict:
#         """Get minimal NPCA statistics"""
#         return {
#             'npca_enabled': self.npca_enabled,
#             'obss_immunity_activations': self.obss_immunity_activations,
#             'vanilla_npca': True  # Flag to indicate this is vanilla implementation
#         }

class NPCASTAFiniteStateMachine(STAFiniteStateMachine):
    """NPCA (Next-generation Primary Channel Access) Station implementation
    
    Key NPCA Features:
    1. OBSS Immunity: Does not freeze backoff during OBSS traffic
    2. Smart Collision Detection: Distinguishes between intra-BSS and OBSS collisions
    3. Adaptive Backoff: Adjusts backoff strategy based on OBSS conditions
    4. Enhanced Channel Access: Improved efficiency in OBSS environments
    """
    
    def __init__(self, sta_id: int, channel_id: int, npca_enabled: bool = True):
        super().__init__(sta_id, channel_id)
        self.npca_enabled = npca_enabled
        
        # NPCA-specific parameters
        self.obss_collision_threshold = 3  # Number of suspected OBSS collisions before adaptation
        self.npca_backoff_multiplier = 0.7  # Reduce backoff when OBSS detected
        self.min_backoff_reduction = 0.5   # Minimum backoff reduction factor
        
        # NPCA statistics
        self.obss_collisions = 0           # Collisions attributed to OBSS
        self.intra_bss_collisions = 0      # Collisions attributed to intra-BSS
        self.npca_backoff_adaptations = 0  # Number of times backoff was adapted
        self.obss_immunity_activations = 0 # Times OBSS immunity was used
        
        # NPCA state tracking
        self.recent_obss_activity = []     # Track recent OBSS activity for smart decisions
        self.obss_activity_window = 100    # Slots to track OBSS activity
        self.suspected_obss_collision_streak = 0
        
    def get_npca_backoff(self, base_backoff: int, obss_present: bool) -> int:
        """Calculate NPCA-adapted backoff value"""
        if not self.npca_enabled:
            return base_backoff
            
        # If OBSS is frequently present, reduce backoff aggressively
        if obss_present or self._is_obss_environment():
            adapted_backoff = int(base_backoff * self.npca_backoff_multiplier)
            adapted_backoff = max(adapted_backoff, int(base_backoff * self.min_backoff_reduction))
            self.npca_backoff_adaptations += 1
            return adapted_backoff
        
        return base_backoff
    
    def get_new_backoff(self, obss_present: bool = False) -> int:
        """Override parent's backoff generation with NPCA logic"""
        cw_index = min(self.backoff_stage, len(CONTENTION_WINDOW) - 1)
        cw = CONTENTION_WINDOW[cw_index]
        base_backoff = np.random.randint(0, cw + 1)
        
        # Apply NPCA adaptation
        return self.get_npca_backoff(base_backoff, obss_present)
    
    def update(self, current_slot: int, channel_busy: bool, obss_busy: bool = False) -> bool:
        """NPCA-enhanced FSM update with OBSS immunity"""
        
        # Track OBSS activity for smart decision making
        self._track_obss_activity(current_slot, obss_busy)
        
        self.tx_attempt = False
        
        # Skip update if currently transmitting and transmission not finished
        if self.state == STAState.TRANSMITTING and self.transmitting_until > current_slot:
            return False
        
        # NPCA-specific deferral counting
        if self.npca_enabled:
            self._npca_count_deferrals(channel_busy, obss_busy)
        else:
            # Use legacy deferral counting
            self._legacy_count_deferrals(channel_busy, obss_busy)
        
        # Main FSM logic with NPCA enhancements
        if self.state == STAState.IDLE:
            self._handle_idle_state(current_slot)
            
        elif self.state == STAState.BACKOFF:
            self._npca_handle_backoff(channel_busy, obss_busy)
            
        elif self.state == STAState.BACKOFF_FROZEN:
            self._npca_handle_backoff_frozen(channel_busy, obss_busy)
            
        elif self.state == STAState.OBSS_FROZEN:
            self._npca_handle_obss_frozen(channel_busy, obss_busy)
            
        elif self.state == STAState.TRANSMITTING:
            self._handle_transmitting(current_slot)
        
        return self.tx_attempt
    
    def _npca_count_deferrals(self, channel_busy: bool, obss_busy: bool):
        """NPCA-specific deferral counting logic"""
        if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
            if channel_busy and not obss_busy:
                self.intra_bss_deferrals += 1
            elif obss_busy and not channel_busy:
                # NPCA: Only count OBSS deferrals when explicitly choosing to defer
                if not self.npca_enabled or self._should_defer_to_obss():
                    self.obss_deferrals += 1
            elif channel_busy and obss_busy:
                self.intra_bss_deferrals += 1  # Intra-BSS takes priority
    
    def _legacy_count_deferrals(self, channel_busy: bool, obss_busy: bool):
        """Legacy deferral counting (same as parent class)"""
        if self.state in [STAState.BACKOFF, STAState.BACKOFF_FROZEN, STAState.OBSS_FROZEN]:
            if channel_busy and not obss_busy:
                self.intra_bss_deferrals += 1
            elif obss_busy and not channel_busy:
                self.obss_deferrals += 1
            elif channel_busy and obss_busy:
                self.intra_bss_deferrals += 1
    
    def _npca_handle_backoff(self, channel_busy: bool, obss_busy: bool):
        """NPCA-enhanced backoff handling with OBSS immunity"""
        if channel_busy:
            # Always defer to intra-BSS traffic
            self.state = STAState.BACKOFF_FROZEN
        elif obss_busy and not self.npca_enabled:
            # Legacy behavior: freeze on OBSS
            self.state = STAState.OBSS_FROZEN
        elif obss_busy and self.npca_enabled:
            # NPCA behavior: immune to OBSS, continue backoff
            self.obss_immunity_activations += 1
            self._proceed_with_backoff()
        else:
            # Channel clear - proceed with backoff
            self._proceed_with_backoff()
    
    def _npca_handle_backoff_frozen(self, channel_busy: bool, obss_busy: bool):
        """NPCA-enhanced frozen state handling"""
        if not channel_busy and not obss_busy:
            # Both cleared - resume backoff
            self.state = STAState.BACKOFF
        elif not channel_busy and obss_busy:
            if self.npca_enabled:
                # NPCA: Resume backoff despite OBSS
                self.state = STAState.BACKOFF
                self.obss_immunity_activations += 1
            else:
                # Legacy: Transition to OBSS_FROZEN
                self.state = STAState.OBSS_FROZEN
    
    def _npca_handle_obss_frozen(self, channel_busy: bool, obss_busy: bool):
        """NPCA-enhanced OBSS frozen state (mainly for legacy mode)"""
        if channel_busy:
            # Intra-BSS appeared - higher priority
            self.state = STAState.BACKOFF_FROZEN
        elif not obss_busy:
            # OBSS cleared - resume backoff
            self.state = STAState.BACKOFF
    
    def _proceed_with_backoff(self):
        """Proceed with backoff countdown"""
        if self.backoff_counter == 0:
            self.state = STAState.TRANSMITTING
            self.tx_attempt = True
            self.total_attempts += 1
        else:
            self.backoff_counter -= 1
            if self.backoff_counter == 0:
                self.state = STAState.TRANSMITTING
                self.tx_attempt = True
                self.total_attempts += 1
    
    def _track_obss_activity(self, current_slot: int, obss_busy: bool):
        """Track OBSS activity for smart decision making"""
        # Add current OBSS status to activity window
        self.recent_obss_activity.append((current_slot, obss_busy))
        
        # Keep only recent activity within window
        cutoff_slot = current_slot - self.obss_activity_window
        self.recent_obss_activity = [
            (slot, busy) for slot, busy in self.recent_obss_activity 
            if slot > cutoff_slot
        ]
    
    def _is_obss_environment(self) -> bool:
        """Determine if we're in an OBSS-heavy environment"""
        if len(self.recent_obss_activity) < 10:
            return False
        
        obss_busy_count = sum(1 for _, busy in self.recent_obss_activity if busy)
        obss_ratio = obss_busy_count / len(self.recent_obss_activity)
        
        return obss_ratio > 0.1  # Consider OBSS environment if >10% of recent slots had OBSS
    
    def _should_defer_to_obss(self) -> bool:
        """NPCA decision: Should we defer to OBSS traffic?"""
        if not self.npca_enabled:
            return True  # Legacy behavior: always defer
        
        # NPCA: Generally don't defer to OBSS, but might in special cases
        # For now, implement aggressive OBSS immunity
        return False
    
    def on_transmission_result(self, result: str, completion_slot: int):
        """Enhanced transmission result handling with NPCA collision analysis"""
        if self.state != STAState.TRANSMITTING:
            return
            
        if result == 'success':
            self.state = STAState.IDLE
            self.successful_transmissions += 1
            self.last_successful_tx_slot = completion_slot
            self.suspected_obss_collision_streak = 0  # Reset streak on success
            self._reset_transmission_params()
            
        elif result == 'collision':
            # NPCA: Analyze collision type
            collision_type = self._analyze_collision_type()
            
            if collision_type == 'obss':
                self.obss_collisions += 1
                self.suspected_obss_collision_streak += 1
            else:
                self.intra_bss_collisions += 1
                self.suspected_obss_collision_streak = 0
            
            # Adjust backoff based on collision analysis
            if self.npca_enabled and collision_type == 'obss':
                # Less aggressive backoff increase for OBSS collisions
                if self.suspected_obss_collision_streak < self.obss_collision_threshold:
                    # Don't increase backoff stage for suspected OBSS collisions
                    pass
                else:
                    # Multiple OBSS collisions - increase stage but less aggressively
                    self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 2)
            else:
                # Normal backoff increase for intra-BSS collisions
                self.backoff_stage = min(self.backoff_stage + 1, len(CONTENTION_WINDOW) - 1)
            
            # Generate new backoff with NPCA adaptation
            obss_present = self._is_obss_environment()
            self.backoff_counter = self.get_new_backoff(obss_present)
            self.collision_count += 1
            self.state = STAState.BACKOFF
            self._reset_transmission_params(keep_frame=True)
    
    def _analyze_collision_type(self) -> str:
        """Analyze whether collision was likely due to OBSS or intra-BSS traffic"""
        if not self.npca_enabled:
            return 'intra_bss'  # Legacy mode doesn't distinguish
        
        # Simple heuristic: if we've been in OBSS environment and had recent OBSS activity,
        # assume collision might be OBSS-related
        if self._is_obss_environment():
            recent_obss_slots = sum(1 for _, busy in self.recent_obss_activity[-10:] if busy)
            if recent_obss_slots > 3:  # Recent OBSS activity
                return 'obss'
        
        return 'intra_bss'
    
    def get_npca_statistics(self) -> Dict:
        """Get NPCA-specific statistics"""
        total_collisions = self.obss_collisions + self.intra_bss_collisions
        
        return {
            'npca_enabled': self.npca_enabled,
            'obss_collisions': self.obss_collisions,
            'intra_bss_collisions': self.intra_bss_collisions,
            'obss_collision_ratio': self.obss_collisions / max(1, total_collisions),
            'npca_backoff_adaptations': self.npca_backoff_adaptations,
            'obss_immunity_activations': self.obss_immunity_activations,
            'obss_environment_detected': self._is_obss_environment(),
            'recent_obss_activity_ratio': (
                sum(1 for _, busy in self.recent_obss_activity if busy) / 
                max(1, len(self.recent_obss_activity))
            ),
            'suspected_obss_collision_streak': self.suspected_obss_collision_streak
        }