import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from random_access.random_access import SimplifiedCSMACASimulation, STAFiniteStateMachine, FrameInfo, SLOTTIME
from random_access.npca_random_access import CrossChannelNPCASTAFiniteStateMachine

class NPCASimulation(SimplifiedCSMACASimulation):
    """Enhanced CSMA/CA Simulation with Cross-Channel NPCA support"""
    
    def __init__(self, num_channels: int, stas_per_channel: list, 
                 simulation_time: int, frame_size: int, 
                 npca_stas_per_channel: list = None,
                 obss_enabled: bool = False,
                 obss_generation_rate: float = 0.001, 
                 obss_frame_size_range: tuple = (20, 50)):
        
        # Initialize parent class first
        super().__init__(
            num_channels=num_channels,
            stas_per_channel=stas_per_channel,
            simulation_time=simulation_time,
            frame_size=frame_size,
            obss_enabled=obss_enabled,
            obss_generation_rate=obss_generation_rate,
            obss_frame_size_range=obss_frame_size_range
        )
        
        # Store NPCA configuration
        if npca_stas_per_channel is None:
            npca_stas_per_channel = [0] * num_channels
        self.npca_stas_per_channel = npca_stas_per_channel
        
        # Setup Cross-Channel OBSS configuration
        self._setup_cross_channel_obss()
        
        # Replace some legacy STAs with Cross-Channel NPCA STAs
        self._convert_to_npca_stas()
        
        # Re-generate initial frames for all stations
        self._generate_initial_frames()
    
    def _setup_cross_channel_obss(self):
        """Setup Cross-Channel OBSS configuration"""
        from random_access.npca_random_access import CrossChannelNPCASimulation
        
        # Create cross-channel manager
        self.cross_channel_manager = CrossChannelNPCASimulation(self)
        
        # Modify OBSS generation: Channel 0 has no self-generated OBSS
        if self.obss_enabled:
            # Keep only Channel 1 OBSS generator
            original_generators = self.obss_generators.copy()
            self.obss_generators = []
            
            # Add OBSS generator only for Channel 1
            for gen in original_generators:
                if gen.source_channel == 1:  # Only Channel 1 generates its own OBSS
                    self.obss_generators.append(gen)
            
            print("🔧 OBSS Configuration:")
            print(f"   Channel 0: No self-generated OBSS (receives NPCA OBSS only)")
            print(f"   Channel 1: Self-generated OBSS at {self.obss_generation_rate:.1%} rate")
    
    def _convert_to_npca_stas(self):
        """Convert specified number of legacy STAs to Cross-Channel NPCA STAs"""
        for ch_id in range(self.num_channels):
            npca_count = self.npca_stas_per_channel[ch_id]
            
            if npca_count <= 0:
                continue
                
            # Find legacy STAs on this channel
            channel_stas = [sta for sta in self.stations if sta.channel_id == ch_id]
            
            # Ensure we don't exceed available STAs
            npca_count = min(npca_count, len(channel_stas))
            
            # Convert first N STAs to Cross-Channel NPCA STAs
            for i in range(npca_count):
                old_sta = channel_stas[i]
                sta_index = self.stations.index(old_sta)
                
                # Determine alternative channel (Channel 1 STAs use Channel 0 as alternative)
                alternative_channel = 0 if ch_id == 1 else 1
                
                # Create new Cross-Channel NPCA STA
                npca_sta = CrossChannelNPCASTAFiniteStateMachine(
                    sta_id=old_sta.sta_id,
                    channel_id=old_sta.channel_id,
                    npca_enabled=True,
                    alternative_channel=alternative_channel
                )
                
                # Set channel trackers for cross-channel operation
                primary_tracker = self.channels[ch_id]
                alternative_tracker = self.channels[alternative_channel]
                npca_sta.set_channel_trackers(primary_tracker, alternative_tracker)
                
                # Replace in stations list
                self.stations[sta_index] = npca_sta
                
                print(f"Converted STA {old_sta.sta_id} on Channel {ch_id} to Cross-Channel NPCA")
                print(f"  Primary: Channel {ch_id}, Alternative: Channel {alternative_channel}")

    def _generate_new_frame(self, sta, creation_slot: int):
        """Generate new frame for station after successful transmission"""
        frame = FrameInfo(
            frame_id=self.frame_counter,
            source=sta.sta_id,
            size=self.frame_size,
            timestamp=creation_slot,
            creation_slot=creation_slot
        )
        
        self.frame_counter += 1
        sta.add_frame(frame)

    def _generate_initial_frames(self):
        """Generate initial frames for all stations"""
        self.frame_counter = 0
        for sta in self.stations:
            frame = FrameInfo(
                frame_id=self.frame_counter,
                source=sta.sta_id,
                size=self.frame_size,
                timestamp=0,
                creation_slot=0
            )
            
            self.frame_counter += 1
            sta.add_frame(frame)

    def _tick(self):
        """Enhanced tick with Cross-Channel NPCA support"""
        # Generate OBSS traffic first (only Channel 1 now)
        self._generate_obss_traffic(self.current_slot)
        
        # Update channels first and get completed transmission results
        completed_results = {}
        for channel in self.channels:
            results = channel.update(self.current_slot)
            if results:
                completed_results[channel.channel_id] = results
        
        # Process completed transmission results
        for ch_id, results in completed_results.items():
            for sta_id, result, completion_slot in results:
                sta = self.stations[sta_id]
                sta.on_transmission_result(result, completion_slot)
                
                # Generate new frame after successful transmission
                if result == 'success':
                    self._generate_new_frame(sta, completion_slot)
        
        # Update stations and collect transmission attempts (with cross-channel support)
        for sta in self.stations:
            if isinstance(sta, CrossChannelNPCASTAFiniteStateMachine):
                # Cross-Channel NPCA STA
                primary_channel = self.channels[sta.primary_channel]
                channel_busy = primary_channel.is_busy(self.current_slot)
                obss_busy = primary_channel.is_obss_busy(self.current_slot)
                
                # Update NPCA STA and get transmission attempt info
                tx_attempt, target_channel = sta.update(self.current_slot, channel_busy, obss_busy)
                
                # Handle transmission attempt
                if tx_attempt and sta.current_frame:
                    target_channel_fsm = self.channels[target_channel]
                    target_channel_fsm.add_transmission(sta.sta_id, sta.current_frame)
                    
                    # If NPCA is transmitting on alternative channel, create OBSS
                    if target_channel != sta.primary_channel:
                        self.cross_channel_manager.handle_npca_transmission(
                            sta, self.current_slot, target_channel
                        )
            else:
                # Legacy STA
                channel = self.channels[sta.channel_id]
                channel_busy = channel.is_busy(self.current_slot)
                obss_busy = channel.is_obss_busy(self.current_slot)
                
                # Update station FSM with separate intra-BSS and OBSS busy signals
                tx_attempt = sta.update(self.current_slot, channel_busy, obss_busy)
                
                # Collect transmission attempts
                if tx_attempt and sta.current_frame:
                    channel.add_transmission(sta.sta_id, sta.current_frame)
        
        # Resolve channel access (schedule future results)
        for channel in self.channels:
            channel.resolve_access(self.current_slot)
        
        # Log current state
        self._log_state()
    
    def get_enhanced_statistics(self):
        """Enhanced statistics with Cross-Channel NPCA metrics"""
        # Get base statistics
        stats = super().get_statistics()
        
        # Add Cross-Channel NPCA-specific statistics
        npca_stats = {
            'npca_stas_per_channel': self.npca_stas_per_channel,
            'total_npca_stas': sum(self.npca_stas_per_channel),
            'total_legacy_stas': len(self.stations) - sum(self.npca_stas_per_channel),
            'npca_stations': {},
            'legacy_stations': {},
            'npca_summary': {
                'total_obss_immunity_activations': 0,
                'total_channel_switches': 0,
                'total_alternative_channel_transmissions': 0,
                'avg_alternative_channel_success_rate': 0.0,
                'cross_channel_npca_used': True
            },
            'cross_channel_obss': {},
            'performance_comparison': {
                'npca_avg_throughput': 0.0,
                'legacy_avg_throughput': 0.0,
                'npca_avg_aoi': 0.0,
                'legacy_avg_aoi': 0.0,
                'throughput_improvement': 0.0,
                'aoi_improvement': 0.0
            },
            'channel_specific_performance': {
                'channel_0_legacy_throughput': 0.0,
                'channel_1_npca_throughput': 0.0,
                'channel_1_legacy_throughput': 0.0,
                'channel_0_npca_obss_impact': 0.0
            }
        }
        
        # Get Cross-Channel OBSS statistics
        if hasattr(self, 'cross_channel_manager'):
            npca_stats['cross_channel_obss'] = self.cross_channel_manager.get_npca_obss_statistics()
        
        # Separate NPCA and Legacy STA statistics
        npca_throughputs = []
        legacy_throughputs = []
        npca_aois = []
        legacy_aois = []
        
        # Channel-specific analysis
        ch0_legacy_throughputs = []
        ch1_npca_throughputs = []
        ch1_legacy_throughputs = []
        
        alternative_success_rates = []
        
        for sta in self.stations:
            sta_stats = stats['stations'][sta.sta_id]
            
            if isinstance(sta, CrossChannelNPCASTAFiniteStateMachine) and sta.npca_enabled:
                # Cross-Channel NPCA STA
                npca_specific = sta.get_npca_statistics()
                sta_stats.update(npca_specific)
                npca_stats['npca_stations'][sta.sta_id] = sta_stats
                
                # Collect performance metrics
                throughput = (sta_stats['successful_transmissions'] * self.frame_size) / stats['total_slots']
                npca_throughputs.append(throughput)
                npca_aois.append(sta_stats['average_aoi_time_us'])
                
                # Channel-specific metrics
                if sta.primary_channel == 1:
                    ch1_npca_throughputs.append(throughput)
                
                # Aggregate Cross-Channel NPCA summary
                npca_stats['npca_summary']['total_obss_immunity_activations'] += npca_specific['obss_immunity_activations']
                npca_stats['npca_summary']['total_channel_switches'] += npca_specific['channel_switches']
                npca_stats['npca_summary']['total_alternative_channel_transmissions'] += npca_specific['alternative_channel_transmissions']
                alternative_success_rates.append(npca_specific['alternative_channel_success_rate'])
                
            else:
                # Legacy STA
                npca_stats['legacy_stations'][sta.sta_id] = sta_stats
                
                # Collect performance metrics
                throughput = (sta_stats['successful_transmissions'] * self.frame_size) / stats['total_slots']
                legacy_throughputs.append(throughput)
                legacy_aois.append(sta_stats['average_aoi_time_us'])
                
                # Channel-specific metrics
                if sta.channel_id == 0:
                    ch0_legacy_throughputs.append(throughput)
                elif sta.channel_id == 1:
                    ch1_legacy_throughputs.append(throughput)
        
        # Calculate performance comparison
        if npca_throughputs:
            npca_stats['performance_comparison']['npca_avg_throughput'] = np.mean(npca_throughputs)
            npca_stats['performance_comparison']['npca_avg_aoi'] = np.mean(npca_aois)
            if alternative_success_rates:
                npca_stats['npca_summary']['avg_alternative_channel_success_rate'] = np.mean(alternative_success_rates)
        
        if legacy_throughputs:
            npca_stats['performance_comparison']['legacy_avg_throughput'] = np.mean(legacy_throughputs)
            npca_stats['performance_comparison']['legacy_avg_aoi'] = np.mean(legacy_aois)
        
        # Channel-specific performance
        if ch0_legacy_throughputs:
            npca_stats['channel_specific_performance']['channel_0_legacy_throughput'] = np.mean(ch0_legacy_throughputs)
        
        if ch1_npca_throughputs:
            npca_stats['channel_specific_performance']['channel_1_npca_throughput'] = np.mean(ch1_npca_throughputs)
        
        if ch1_legacy_throughputs:
            npca_stats['channel_specific_performance']['channel_1_legacy_throughput'] = np.mean(ch1_legacy_throughputs)
        
        # Calculate improvements
        if legacy_throughputs and npca_throughputs:
            legacy_avg = npca_stats['performance_comparison']['legacy_avg_throughput']
            npca_avg = npca_stats['performance_comparison']['npca_avg_throughput']
            npca_stats['performance_comparison']['throughput_improvement'] = ((npca_avg - legacy_avg) / legacy_avg * 100) if legacy_avg > 0 else 0
            
            legacy_aoi_avg = npca_stats['performance_comparison']['legacy_avg_aoi']
            npca_aoi_avg = npca_stats['performance_comparison']['npca_avg_aoi']
            npca_stats['performance_comparison']['aoi_improvement'] = ((legacy_aoi_avg - npca_aoi_avg) / legacy_aoi_avg * 100) if legacy_aoi_avg > 0 else 0
        
        # Merge with base statistics
        stats.update(npca_stats)
        
        return stats
    
    def _log_state(self):
        """Enhanced state logging with Cross-Channel NPCA information"""
        # Get base log entry
        super()._log_state()
        
        # Add Cross-Channel NPCA-specific logging to the last log entry
        log_entry = self.logs[-1] if self.logs else {}
        
        # Add Cross-Channel NPCA states by channel
        for ch_id in range(self.num_channels):
            channel_stas = [sta for sta in self.stations if sta.channel_id == ch_id]
            
            npca_enabled_list = []
            channel_switches = []
            current_active_channels = []
            
            for sta in channel_stas:
                if isinstance(sta, CrossChannelNPCASTAFiniteStateMachine):
                    npca_enabled_list.append(sta.npca_enabled)
                    channel_switches.append(sta.channel_switches)
                    current_active_channels.append(sta.current_active_channel)
                else:
                    npca_enabled_list.append(False)
                    channel_switches.append(0)
                    current_active_channels.append(sta.channel_id)
            
            log_entry[f'npca_enabled_ch_{ch_id}'] = npca_enabled_list
            log_entry[f'channel_switches_ch_{ch_id}'] = channel_switches
            log_entry[f'current_active_channels_ch_{ch_id}'] = current_active_channels


# Test configuration for Cross-Channel NPCA scenarios
def create_npca_test_configs():
    """Create test configurations for Cross-Channel NPCA evaluation"""
    
    simulation_time = 50000  # Reduced for faster testing
    frame_size = 33
    
    configs = [
        # Baseline: No OBSS, All Legacy
        {
            "num_channels": 2, 
            "stas_per_channel": [2, 10], 
            "npca_stas_per_channel": [0, 0],  # All legacy
            "obss_enabled": False,
            "obss_generation_rate": 0.0,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "Baseline (No OBSS, All Legacy)"
        },
        
        # High OBSS: All Legacy
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 0],  # All legacy
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS (All Legacy)"
        },
        
        # High OBSS: 1 NPCA + 9 Legacy on Channel 1
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 1],  # 1 NPCA on channel 1
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS (1 NPCA + 9 Legacy)"
        },
        
        # High OBSS: 5 NPCA + 5 Legacy on Channel 1
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 5],  # 5 NPCA on channel 1
            "obss_enabled": True,
            "obss_generation_rate": 0.002,
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "High OBSS (5 NPCA + 5 Legacy)"
        },
        
        # Extremely High OBSS: 5 NPCA + 5 Legacy on Channel 1
        {
            "num_channels": 2,
            "stas_per_channel": [2, 10],
            "npca_stas_per_channel": [0, 5],  # 5 NPCA on channel 1
            "obss_enabled": True,
            "obss_generation_rate": 0.02,  # 2% OBSS rate
            "simulation_time": simulation_time,
            "frame_size": frame_size,
            "label": "Extremely High OBSS (5 NPCA + 5 Legacy)"
        }
    ]
    
    return configs


def run_npca_test():
    """Quick test function to verify Cross-Channel NPCA implementation"""
    print("🧪 Running Cross-Channel NPCA Test...")
    
    # Simple test configuration
    config = {
        "num_channels": 2,
        "stas_per_channel": [1, 4],  # 1 STA on ch0, 4 STAs on ch1
        "npca_stas_per_channel": [0, 2],  # 2 NPCA STAs on ch1
        "obss_enabled": True,
        "obss_generation_rate": 0.01,  # 1% OBSS rate
        "simulation_time": 10000,
        "frame_size": 33
    }
    
    print(f"Test config: {config['stas_per_channel']} STAs per channel")
    print(f"NPCA STAs: {config['npca_stas_per_channel']} per channel")
    print(f"OBSS rate: {config['obss_generation_rate']:.1%}")
    
    # Run simulation
    sim = NPCASimulation(
        num_channels=config["num_channels"],
        stas_per_channel=config["stas_per_channel"],
        npca_stas_per_channel=config["npca_stas_per_channel"],
        simulation_time=config["simulation_time"],
        frame_size=config["frame_size"],
        obss_enabled=config["obss_enabled"],
        obss_generation_rate=config["obss_generation_rate"]
    )
    
    # Run simulation
    df = sim.run()
    stats = sim.get_enhanced_statistics()
    
    # Print test results
    print(f"\n📊 Test Results:")
    print(f"Total simulation slots: {stats['total_slots']}")
    print(f"NPCA STAs: {stats['total_npca_stas']}")
    print(f"Legacy STAs: {stats['total_legacy_stas']}")
    print(f"OBSS events generated: {stats['obss_events_generated']}")
    
    print(f"\n🎯 Performance Comparison:")
    perf = stats['performance_comparison']
    print(f"NPCA avg throughput: {perf['npca_avg_throughput']:.6f}")
    print(f"Legacy avg throughput: {perf['legacy_avg_throughput']:.6f}")
    print(f"Throughput improvement: {perf['throughput_improvement']:.2f}%")
    print(f"AoI improvement: {perf['aoi_improvement']:.2f}%")
    
    print(f"\n⚡ Cross-Channel NPCA Summary:")
    npca_sum = stats['npca_summary']
    print(f"OBSS immunity activations: {npca_sum['total_obss_immunity_activations']}")
    print(f"Channel switches: {npca_sum['total_channel_switches']}")
    print(f"Alternative channel transmissions: {npca_sum['total_alternative_channel_transmissions']}")
    print(f"Alternative channel success rate: {npca_sum['avg_alternative_channel_success_rate']:.3f}")
    
    # Cross-Channel OBSS impact
    if 'cross_channel_obss' in stats:
        cc_obss = stats['cross_channel_obss']
        print(f"\n🔄 Cross-Channel OBSS Impact:")
        print(f"NPCA OBSS events: {cc_obss.get('total_npca_obss_events', 0)}")
        print(f"Channel 0 NPCA OBSS events: {cc_obss.get('channel_0_npca_obss_events', 0)}")
        print(f"Channel 0 NPCA OBSS duration: {cc_obss.get('channel_0_npca_obss_duration', 0)} slots")
    
    print("\n✅ Cross-Channel NPCA Test Complete!")
    return sim, df, stats


if __name__ == "__main__":
    # Run quick test
    sim, df, stats = run_npca_test()