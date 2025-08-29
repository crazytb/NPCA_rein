# npca_log_analyzer.py
"""
Comprehensive NPCA Performance Log Analyzer
Analyzes detailed logs to understand why NPCA performance is degraded
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class NPCALogAnalyzer:
    """Analyze NPCA simulation logs in detail"""
    
    def __init__(self, df: pd.DataFrame, stats: Dict, config: Dict):
        self.df = df
        self.stats = stats
        self.config = config
        
        # Extract channel information
        self.num_channels = config.get('num_channels', 2)
        self.frame_size = config.get('frame_size', 33)
        
        print(f"📊 Analyzing {len(df)} simulation slots...")
        print(f"   Channels: {self.num_channels}")
        print(f"   NPCA STAs: {stats.get('total_npca_stas', 0)}")
        print(f"   Legacy STAs: {stats.get('total_legacy_stas', 0)}")
    
    def analyze_obss_impact(self):
        """Analyze OBSS traffic impact on different STA types"""
        print("\n🔍 OBSS TRAFFIC IMPACT ANALYSIS")
        print("=" * 50)
        
        # OBSS activity analysis
        obss_slots = 0
        intra_bss_slots = 0
        idle_slots = 0
        
        for ch_id in range(self.num_channels):
            if f'channel_{ch_id}_obss_busy' in self.df.columns:
                obss_busy = self.df[f'channel_{ch_id}_obss_busy']
                intra_busy = self.df[f'channel_{ch_id}_busy']
                
                obss_only = obss_busy & ~intra_busy
                intra_only = intra_busy & ~obss_busy
                both_busy = obss_busy & intra_busy
                idle = ~obss_busy & ~intra_busy
                
                print(f"\n📡 Channel {ch_id} Activity:")
                print(f"   OBSS only: {obss_only.sum()} slots ({obss_only.mean()*100:.2f}%)")
                print(f"   Intra-BSS only: {intra_only.sum()} slots ({intra_only.mean()*100:.2f}%)")
                print(f"   Both busy: {both_busy.sum()} slots ({both_busy.mean()*100:.2f}%)")
                print(f"   Idle: {idle.sum()} slots ({idle.mean()*100:.2f}%)")
    
    def analyze_sta_behavior(self):
        """Analyze individual STA behavior patterns"""
        print("\n👥 INDIVIDUAL STA BEHAVIOR ANALYSIS")
        print("=" * 50)
        
        for ch_id in range(self.num_channels):
            if f'states_ch_{ch_id}' not in self.df.columns:
                continue
                
            print(f"\n📊 Channel {ch_id} STA Analysis:")
            
            # Get STA information
            npca_enabled = self.df[f'npca_enabled_ch_{ch_id}'].iloc[0] if f'npca_enabled_ch_{ch_id}' in self.df.columns else []
            states = self.df[f'states_ch_{ch_id}']
            tx_attempts = self.df[f'tx_attempts_ch_{ch_id}']
            
            num_stas = len(npca_enabled) if npca_enabled else 0
            if num_stas == 0:
                continue
            
            for sta_idx in range(num_stas):
                is_npca = npca_enabled[sta_idx] if sta_idx < len(npca_enabled) else False
                sta_type = "NPCA" if is_npca else "Legacy"
                
                # Extract this STA's behavior
                sta_states = [slot_states[sta_idx] if sta_idx < len(slot_states) else 'idle' 
                             for slot_states in states]
                sta_tx_attempts = [slot_attempts[sta_idx] if sta_idx < len(slot_attempts) else False 
                                  for slot_attempts in tx_attempts]
                
                # State distribution
                state_counts = {}
                for state in sta_states:
                    state_counts[state] = state_counts.get(state, 0) + 1
                
                # Transmission attempts
                total_attempts = sum(sta_tx_attempts)
                
                print(f"   STA {sta_idx} ({sta_type}):")
                print(f"     States: {dict(sorted(state_counts.items()))}")
                print(f"     TX attempts: {total_attempts}")
                
                # Calculate state percentages
                total_slots = len(sta_states)
                if total_slots > 0:
                    print(f"     State %: ", end="")
                    for state, count in sorted(state_counts.items()):
                        pct = count / total_slots * 100
                        print(f"{state}={pct:.1f}% ", end="")
                    print()
    
    def analyze_npca_mechanisms(self):
        """Analyze NPCA-specific mechanism usage"""
        print("\n⚡ NPCA MECHANISM ANALYSIS")
        print("=" * 50)
        
        if 'npca_summary' not in self.stats:
            print("No NPCA summary data available")
            return
        
        npca_summary = self.stats['npca_summary']
        
        print(f"OBSS Immunity Activations: {npca_summary.get('total_obss_immunity_activations', 0)}")
        print(f"Frame Adaptations: {npca_summary.get('total_frame_adaptations', 0)}")
        print(f"Truncated Frames: {npca_summary.get('total_truncated_frames', 0)}")
        print(f"Frame Efficiency: {npca_summary.get('avg_frame_efficiency', 0):.3f}")
        
        # Analyze frame adaptation patterns
        if 'current_frame_sizes_ch_1' in self.df.columns:
            frame_sizes = self.df['current_frame_sizes_ch_1']
            print(f"\n📏 Frame Size Analysis (Channel 1):")
            
            for slot_idx, slot_sizes in enumerate(frame_sizes[:100]):  # First 100 slots
                if any(size != self.frame_size for size in slot_sizes):
                    print(f"   Slot {slot_idx}: {slot_sizes}")
    
    def analyze_collision_patterns(self):
        """Analyze collision patterns between NPCA and Legacy STAs"""
        print("\n💥 COLLISION PATTERN ANALYSIS")  
        print("=" * 50)
        
        total_npca_collisions = 0
        total_legacy_collisions = 0
        total_npca_success = 0
        total_legacy_success = 0
        
        # Get individual STA statistics
        if 'npca_stations' in self.stats:
            for sta_id, sta_stats in self.stats['npca_stations'].items():
                total_npca_collisions += sta_stats.get('collisions', 0)
                total_npca_success += sta_stats.get('successful_transmissions', 0)
        
        if 'legacy_stations' in self.stats:
            for sta_id, sta_stats in self.stats['legacy_stations'].items():
                total_legacy_collisions += sta_stats.get('collisions', 0)
                total_legacy_success += sta_stats.get('successful_transmissions', 0)
        
        print(f"NPCA STAs:")
        print(f"   Successful transmissions: {total_npca_success}")
        print(f"   Collisions: {total_npca_collisions}")
        npca_success_rate = total_npca_success / max(1, total_npca_success + total_npca_collisions)
        print(f"   Success rate: {npca_success_rate:.3f}")
        
        print(f"\nLegacy STAs:")
        print(f"   Successful transmissions: {total_legacy_success}")
        print(f"   Collisions: {total_legacy_collisions}")
        legacy_success_rate = total_legacy_success / max(1, total_legacy_success + total_legacy_collisions)
        print(f"   Success rate: {legacy_success_rate:.3f}")
        
        print(f"\n📊 Comparison:")
        print(f"   NPCA success rate: {npca_success_rate:.3f}")
        print(f"   Legacy success rate: {legacy_success_rate:.3f}")
        print(f"   Success rate difference: {npca_success_rate - legacy_success_rate:+.3f}")
    
    def analyze_timing_conflicts(self):
        """Analyze timing conflicts between NPCA and other traffic"""
        print("\n⏰ TIMING CONFLICT ANALYSIS")
        print("=" * 50)
        
        # Look for simultaneous transmissions
        conflict_slots = []
        
        for ch_id in range(self.num_channels):
            if f'tx_attempts_ch_{ch_id}' not in self.df.columns:
                continue
                
            tx_attempts = self.df[f'tx_attempts_ch_{ch_id}']
            obss_busy = self.df[f'channel_{ch_id}_obss_busy'] if f'channel_{ch_id}_obss_busy' in self.df.columns else [False] * len(tx_attempts)
            intra_busy = self.df[f'channel_{ch_id}_busy'] if f'channel_{ch_id}_busy' in self.df.columns else [False] * len(tx_attempts)
            
            for slot_idx, (attempts, obss, intra) in enumerate(zip(tx_attempts, obss_busy, intra_busy)):
                if len(attempts) > 0 and any(attempts):  # Someone is transmitting
                    num_transmitters = sum(attempts)
                    
                    if num_transmitters > 1:  # Multiple transmitters = collision
                        conflict_slots.append({
                            'slot': slot_idx,
                            'channel': ch_id,
                            'transmitters': num_transmitters,
                            'obss_busy': obss,
                            'intra_busy': intra
                        })
        
        print(f"Total conflict slots found: {len(conflict_slots)}")
        
        if len(conflict_slots) > 0:
            # Show first few conflicts
            print(f"\nFirst 10 conflicts:")
            for i, conflict in enumerate(conflict_slots[:10]):
                print(f"   Slot {conflict['slot']}: {conflict['transmitters']} transmitters, "
                      f"OBSS={conflict['obss_busy']}, Intra={conflict['intra_busy']}")
            
            # Analyze conflict patterns
            obss_conflicts = sum(1 for c in conflict_slots if c['obss_busy'])
            intra_conflicts = sum(1 for c in conflict_slots if c['intra_busy'])
            
            print(f"\nConflict breakdown:")
            print(f"   During OBSS: {obss_conflicts} ({obss_conflicts/len(conflict_slots)*100:.1f}%)")
            print(f"   During Intra-BSS: {intra_conflicts} ({intra_conflicts/len(conflict_slots)*100:.1f}%)")
    
    def analyze_throughput_degradation(self):
        """Analyze why throughput is degraded"""
        print("\n📉 THROUGHPUT DEGRADATION ANALYSIS")
        print("=" * 50)
        
        perf = self.stats.get('performance_comparison', {})
        
        npca_throughput = perf.get('npca_avg_throughput', 0)
        legacy_throughput = perf.get('legacy_avg_throughput', 0)
        improvement = perf.get('throughput_improvement', 0)
        
        print(f"NPCA Average Throughput: {npca_throughput:.6f}")
        print(f"Legacy Average Throughput: {legacy_throughput:.6f}")
        print(f"Improvement: {improvement:+.2f}%")
        
        # Calculate theoretical maximum
        total_slots = len(self.df)
        max_possible_throughput = self.frame_size / total_slots  # If every slot had successful transmission
        
        print(f"\nTheoretical Analysis:")
        print(f"   Total slots: {total_slots}")
        print(f"   Frame size: {self.frame_size}")
        print(f"   Max possible throughput per STA: {max_possible_throughput:.6f}")
        
        # Analyze utilization
        total_npca_stas = self.stats.get('total_npca_stas', 0)
        total_legacy_stas = self.stats.get('total_legacy_stas', 0)
        
        if total_npca_stas > 0:
            npca_utilization = npca_throughput / max_possible_throughput
            print(f"   NPCA utilization: {npca_utilization:.3f} ({npca_utilization*100:.1f}%)")
        
        if total_legacy_stas > 0:
            legacy_utilization = legacy_throughput / max_possible_throughput
            print(f"   Legacy utilization: {legacy_utilization:.3f} ({legacy_utilization*100:.1f}%)")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("📋 NPCA PERFORMANCE SUMMARY REPORT")
        print("="*60)
        
        # Basic info
        print(f"Simulation: {len(self.df)} slots, {self.stats.get('total_npca_stas', 0)} NPCA + {self.stats.get('total_legacy_stas', 0)} Legacy STAs")
        print(f"OBSS Events: {self.stats.get('obss_events_generated', 0)}")
        print(f"OBSS Rate: {self.config.get('obss_generation_rate', 0):.1%}")
        
        # Performance metrics
        perf = self.stats.get('performance_comparison', {})
        print(f"\n🎯 Performance Results:")
        print(f"   Throughput Change: {perf.get('throughput_improvement', 0):+.2f}%")
        print(f"   AoI Change: {perf.get('aoi_improvement', 0):+.2f}%")
        
        # NPCA mechanisms
        if 'npca_summary' in self.stats:
            npca_sum = self.stats['npca_summary']
            print(f"\n⚡ NPCA Mechanisms:")
            print(f"   OBSS Immunity: {npca_sum.get('total_obss_immunity_activations', 0)} times")
            print(f"   Frame Adaptations: {npca_sum.get('total_frame_adaptations', 0)} times")
            print(f"   Frame Efficiency: {npca_sum.get('avg_frame_efficiency', 0):.3f}")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        if perf.get('throughput_improvement', 0) < 0:
            print("   ❌ NPCA showing negative performance impact")
            print("   🔍 Possible causes:")
            print("     - Network too small for NPCA benefits")
            print("     - OBSS rate too low for immunity to matter")
            print("     - Increased collisions due to aggressive behavior")
            print("     - Legacy STAs suffering from increased competition")
        else:
            print("   ✅ NPCA showing positive performance impact")

def analyze_npca_logs(sim, df, stats):
    """Main function to analyze NPCA simulation logs"""
    config = {
        'num_channels': sim.num_channels,
        'frame_size': sim.frame_size,
        'obss_generation_rate': sim.obss_generation_rate,
        'npca_stas_per_channel': sim.npca_stas_per_channel
    }
    
    analyzer = NPCALogAnalyzer(df, stats, config)
    
    # Run all analyses
    analyzer.analyze_obss_impact()
    analyzer.analyze_sta_behavior()
    analyzer.analyze_npca_mechanisms()
    analyzer.analyze_collision_patterns()
    analyzer.analyze_timing_conflicts()
    analyzer.analyze_throughput_degradation()
    analyzer.generate_summary_report()
    
    return analyzer