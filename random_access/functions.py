import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_access.random_access import SimplifiedCSMACASimulation, SLOTTIME
from random_access.configs import *

def run_obss_comparison(simulation_configs, saveformat="csv"):
    """Run simulations comparing OBSS enabled vs disabled with channel-specific settings"""
    
    results = {}
    
    for config in simulation_configs:
        print(f"Running simulation: {config['label']}")
        
        sim = SimplifiedCSMACASimulation(
            num_channels=config["num_channels"],
            stas_per_channel=config["stas_per_channel"],
            simulation_time=config["simulation_time"],
            frame_size=config["frame_size"],
            obss_enabled_per_channel=config["obss_enabled_per_channel"],
            npca_enabled=config.get("npca_enabled", None),  # 새로 추가 - 없으면 None
            obss_generation_rate=config["obss_generation_rate"],
            obss_frame_size_range=config["obss_frame_size_range"]
        )
        
        df = sim.run()

        # Save the full DataFrame for detailed analysis
        # df.to_csv(f"csv/obss_simulation_{config['label'].replace(' ', '_').lower()}.csv", index=False)

        # Save simplified version of the DataFrame for easier access
        simplified_columns = [
            'time', 'slot',
            'channel_0_occupied_remained', 'channel_0_obss_occupied_remained', 'states_ch_0', 'backoff_ch_0',
            'channel_1_occupied_remained', 'channel_1_obss_occupied_remained', 'states_ch_1', 'backoff_ch_1',
            'npca_attempts_ch_1', 'npca_successful_ch_1', 'npca_blocked_ch_1', 'npca_enabled_ch_1',
        ]
        
        if saveformat == "csv":
            df[simplified_columns].to_csv(f"csv/obss_simulation_{config['label'].replace(' ', '_').lower()}_simplified.csv", index=False)
        elif saveformat == "pickle":
            df[simplified_columns].to_pickle(f"pickle/obss_simulation_{config['label'].replace(' ', '_').lower()}.pkl")

        # stats = sim.get_statistics()
        
        # results[config['label']] = {
        #     'config': config,
        #     'stats': stats,
        #     'dataframe': df
        # }
        del df

    # return results

def plot_obss_comparison(results):
    """Create OBSS and NPCA comparison plots with channel-specific analysis"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: Channel-specific Throughput Comparison
    ax1 = axes[0, 0]
    
    # Calculate channel-specific throughputs
    max_channels = max(len(results[name]['stats']['obss_enabled_per_channel']) for name in config_names)
    channel_throughputs = {ch_id: [] for ch_id in range(max_channels)}
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        for ch_id in range(max_channels):
            # Find STAs in this channel
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                ch_throughput = (ch_successful * config['frame_size']) / stats['total_slots']
                channel_throughputs[ch_id].append(ch_throughput)
            else:
                channel_throughputs[ch_id].append(0)
    
    # Create grouped bar chart
    x = np.arange(len(config_names))
    width = 0.35
    
    bars_ch0 = ax1.bar(x - width/2, channel_throughputs[0], width, alpha=0.7, 
                       color='skyblue', label='Channel 0')
    bars_ch1 = ax1.bar(x + width/2, channel_throughputs[1], width, alpha=0.7, 
                       color='lightcoral', label='Channel 1')
    
    ax1.set_title('Channel-specific Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Throughput (fraction)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate max for y-axis limit
    all_throughputs = channel_throughputs[0] + channel_throughputs[1]
    ax1.set_ylim(0, max(all_throughputs) * 1.2)
    
    # Add value labels
    for bar, value in zip(bars_ch0, channel_throughputs[0]):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_throughputs)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, value in zip(bars_ch1, channel_throughputs[1]):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_throughputs)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add OBSS/NPCA status indicators for each channel
    for i, config_name in enumerate(config_names):
        obss_config = results[config_name]['stats']['obss_enabled_per_channel']
        npca_config = results[config_name]['stats'].get('npca_enabled_per_channel', [False] * len(obss_config))
        
        # Channel 0 status
        ch0_status = []
        if obss_config[0]:
            ch0_status.append("OBSS")
        if npca_config[0]:
            ch0_status.append("NPCA")
        ch0_text = "/".join(ch0_status) if ch0_status else "Clean"
        
        # Channel 1 status  
        ch1_status = []
        if obss_config[1]:
            ch1_status.append("OBSS")
        if npca_config[1]:
            ch1_status.append("NPCA")
        ch1_text = "/".join(ch1_status) if ch1_status else "Clean"
        
        # Add status text below bars
        ax1.text(x[i] - width/2, -max(all_throughputs)*0.08, ch0_text, 
                ha='center', va='top', fontsize=8, style='italic', color='blue')
        ax1.text(x[i] + width/2, -max(all_throughputs)*0.08, ch1_text, 
                ha='center', va='top', fontsize=8, style='italic', color='red')
    
    # Plot 2: Average AoI Comparison
    ax2 = axes[0, 1]
    avg_aois = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        all_aois = [sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()]
        avg_aoi = np.mean(all_aois)
        avg_aois.append(avg_aoi)
    
    bars2 = ax2.bar(range(len(config_names)), avg_aois, alpha=0.7, color=colors)
    ax2.set_title('Average AoI Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average AoI (μs)', fontsize=12)
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, avg_aois):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_aois)*0.02,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: NPCA Performance Analysis
    ax3 = axes[1, 0]
    
    npca_attempts = []
    npca_successful = []
    npca_success_rates = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        attempts = stats.get('npca_total_attempts', 0)
        successful = stats.get('npca_total_successful', 0)
        success_rate = stats.get('npca_success_rate', 0)
        
        npca_attempts.append(attempts)
        npca_successful.append(successful)
        npca_success_rates.append(success_rate)
    
    x = np.arange(len(config_names))
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, npca_attempts, width, alpha=0.7, 
                     color='lightblue', label='NPCA Attempts')
    bars3b = ax3.bar(x + width/2, npca_successful, width, alpha=0.7, 
                     color='lightgreen', label='NPCA Successful')
    
    ax3.set_title('NPCA Performance Analysis', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels and success rates
    for i, (bar_a, bar_s, attempts, successful, rate) in enumerate(zip(bars3a, bars3b, npca_attempts, npca_successful, npca_success_rates)):
        if attempts > 0:
            ax3.text(bar_a.get_x() + bar_a.get_width()/2, bar_a.get_height() + max(npca_attempts)*0.02,
                    f'{attempts}', ha='center', va='bottom', fontsize=9)
        if successful > 0:
            ax3.text(bar_s.get_x() + bar_s.get_width()/2, bar_s.get_height() + max(npca_attempts)*0.02,
                    f'{successful}', ha='center', va='bottom', fontsize=9)
        
        # Add success rate below
        if attempts > 0:
            ax3.text(x[i], -max(npca_attempts)*0.1, f'{rate:.1f}%', 
                    ha='center', va='top', fontsize=8, fontweight='bold', color='red')
    
    # Plot 4: Channel-specific Traffic Generation (OBSS + NPCA)
    ax4 = axes[1, 1]
    
    # Prepare data for stacked bar chart showing both OBSS events and NPCA events per channel
    max_channels = max(len(results[name]['stats']['obss_enabled_per_channel']) for name in config_names)
    
    # OBSS events data
    obss_data = {}
    npca_data = {}
    
    for ch_id in range(max_channels):
        obss_data[f'Ch{ch_id}_OBSS'] = []
        npca_data[f'Ch{ch_id}_NPCA'] = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            
            # OBSS events
            if ch_id < len(stats['obss_enabled_per_channel']):
                ch_obss_stats = stats['obss_per_channel'][ch_id]
                obss_data[f'Ch{ch_id}_OBSS'].append(ch_obss_stats['generated'])
            else:
                obss_data[f'Ch{ch_id}_OBSS'].append(0)
            
            # NPCA events (from target channel perspective)
            if ch_id < len(stats.get('npca_enabled_per_channel', [])):
                ch_npca_stats = stats.get('npca_per_channel', {}).get(ch_id, {'total_successful': 0})
                npca_data[f'Ch{ch_id}_NPCA'].append(ch_npca_stats['total_successful'])
            else:
                npca_data[f'Ch{ch_id}_NPCA'].append(0)
    
    # Create grouped bar chart
    x = np.arange(len(config_names))
    width = 0.8 / (max_channels * 2)  # Two types per channel
    
    colors_ch = plt.cm.tab10(np.linspace(0, 1, max_channels))
    
    for ch_id in range(max_channels):
        # OBSS bars
        obss_values = obss_data[f'Ch{ch_id}_OBSS']
        npca_values = npca_data[f'Ch{ch_id}_NPCA']
        
        offset_obss = (ch_id * 2 - max_channels + 0.5) * width
        offset_npca = ((ch_id * 2 + 1) - max_channels + 0.5) * width
        
        bars_obss = ax4.bar(x + offset_obss, obss_values, width, alpha=0.7, 
                           color=colors_ch[ch_id], label=f'Ch{ch_id} OBSS' if ch_id == 0 else "")
        bars_npca = ax4.bar(x + offset_npca, npca_values, width, alpha=0.7, 
                           color=colors_ch[ch_id], hatch='///', label=f'Ch{ch_id} NPCA' if ch_id == 0 else "")
        
        # Add value labels for non-zero values
        for bar, value in zip(bars_obss, obss_values):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{int(value)}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars_npca, npca_values):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{int(value)}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_title('Traffic Generation by Channel (OBSS vs NPCA)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Events Generated/Successful', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figure/obss_npca_comparison_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_channel_specific_obss_impact(results):
    """Plot OBSS and NPCA impact on individual channels with detailed analysis"""
    
    config_names = list(results.keys())
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    fig, axes = plt.subplots(num_channels, 4, figsize=(20, 6*num_channels))
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # For each channel, plot throughput, AoI, OBSS impact, and NPCA analysis
    for ch_id in range(num_channels):
        
        # 1. Channel throughput comparison
        ax_throughput = axes[ch_id, 0]
        channel_throughputs = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            config = results[config_name]['config']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            channel_throughput = (total_successful * config['frame_size']) / stats['total_slots']
            channel_throughputs.append(channel_throughput)
        
        bars = ax_throughput.bar(range(len(config_names)), channel_throughputs, alpha=0.7, color=colors)
        ax_throughput.set_title(f'Channel {ch_id} Throughput', fontsize=12, fontweight='bold')
        ax_throughput.set_ylabel('Throughput', fontsize=10)
        ax_throughput.set_xticks(range(len(config_names)))
        ax_throughput.set_xticklabels(config_names, rotation=45, ha='right')
        ax_throughput.grid(True, alpha=0.3)
        
        # Add value labels and status indicators
        for i, (bar, value, config_name) in enumerate(zip(bars, channel_throughputs, config_names)):
            ax_throughput.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(channel_throughputs)*0.02,
                              f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # OBSS and NPCA status for this channel
            obss_enabled = results[config_name]['stats']['obss_enabled_per_channel'][ch_id]
            npca_enabled = results[config_name]['stats'].get('npca_enabled_per_channel', [False]*num_channels)[ch_id]
            
            status_text = ""
            if obss_enabled:
                status_text += "O"
            else:
                status_text += "X"
            
            if npca_enabled:
                status_text += "N"
            else:
                status_text += "-"
            
            # 색상으로 구분
            obss_color = 'green' if obss_enabled else 'red'
            npca_color = 'blue' if npca_enabled else 'gray'
            
            ax_throughput.text(bar.get_x() + bar.get_width()/2, -max(channel_throughputs)*0.08,
                              status_text, ha='center', va='top', fontsize=10, 
                              color=obss_color, fontweight='bold')
        
        # 2. Channel AoI comparison
        ax_aoi = axes[ch_id, 1]
        channel_aois = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                channel_aois.append(ch_aoi)
            else:
                channel_aois.append(0)
        
        bars = ax_aoi.bar(range(len(config_names)), channel_aois, alpha=0.7, color=colors)
        ax_aoi.set_title(f'Channel {ch_id} Avg AoI', fontsize=12, fontweight='bold')
        ax_aoi.set_ylabel('AoI (μs)', fontsize=10)
        ax_aoi.set_xticks(range(len(config_names)))
        ax_aoi.set_xticklabels(config_names, rotation=45, ha='right')
        ax_aoi.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, channel_aois):
            if value > 0:
                ax_aoi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(channel_aois)*0.02,
                           f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. OBSS Impact Analysis
        ax_obss = axes[ch_id, 2]
        
        obss_events = []
        obss_deferrals = []
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            
            # OBSS events generated in this channel
            ch_obss_stats = stats['obss_per_channel'][ch_id]
            obss_events.append(ch_obss_stats['generated'])
            
            # OBSS deferrals experienced by STAs in this channel
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            total_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
            obss_deferrals.append(total_deferrals)
        
        x = np.arange(len(config_names))
        width = 0.35
        
        bars1 = ax_obss.bar(x - width/2, obss_events, width, alpha=0.7, 
                           color='lightblue', label='OBSS Events Generated')
        bars2 = ax_obss.bar(x + width/2, obss_deferrals, width, alpha=0.7, 
                           color='lightcoral', label='STA OBSS Deferrals')
        
        ax_obss.set_title(f'Channel {ch_id} OBSS Impact', fontsize=12, fontweight='bold')
        ax_obss.set_ylabel('Count', fontsize=10)
        ax_obss.set_xticks(x)
        ax_obss.set_xticklabels(config_names, rotation=45, ha='right')
        ax_obss.legend()
        ax_obss.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax_obss.text(bar.get_x() + bar.get_width()/2., height + 5,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 4. NPCA Analysis (New)
        ax_npca = axes[ch_id, 3]
        
        npca_attempts = []
        npca_successful = []
        npca_as_target = []  # This channel used as NPCA target
        
        for config_name in config_names:
            stats = results[config_name]['stats']
            
            # NPCA attempts/successful from STAs in this channel
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            ch_npca_attempts = sum(sta_stats.get('npca_attempts', 0) for _, sta_stats in channel_stas)
            ch_npca_successful = sum(sta_stats.get('npca_successful', 0) for _, sta_stats in channel_stas)
            
            npca_attempts.append(ch_npca_attempts)
            npca_successful.append(ch_npca_successful)
            
            # This channel used as NPCA target (count OBSS traffic from other channels)
            # Rough estimation: count non-local OBSS traffic
            npca_target_count = 0
            for obss_traffic in results[config_name]['dataframe'].get(f'channel_{ch_id}_active_obss_count', []):
                # This is a simplified estimation - in practice, we'd need to track NPCA origins
                pass
            npca_as_target.append(npca_target_count)
        
        x = np.arange(len(config_names))
        width = 0.25
        
        bars1 = ax_npca.bar(x - width, npca_attempts, width, alpha=0.7, 
                           color='lightgreen', label='NPCA Attempts')
        bars2 = ax_npca.bar(x, npca_successful, width, alpha=0.7, 
                           color='darkgreen', label='NPCA Successful')
        bars3 = ax_npca.bar(x + width, npca_as_target, width, alpha=0.7, 
                           color='orange', label='Used as NPCA Target')
        
        ax_npca.set_title(f'Channel {ch_id} NPCA Analysis', fontsize=12, fontweight='bold')
        ax_npca.set_ylabel('Count', fontsize=10)
        ax_npca.set_xticks(x)
        ax_npca.set_xticklabels(config_names, rotation=45, ha='right')
        ax_npca.legend()
        ax_npca.grid(True, alpha=0.3)
        
        # Add value labels and success rates
        max_npca_value = max(max(npca_attempts) if npca_attempts else 0, 
                            max(npca_successful) if npca_successful else 0)
        
        for i, (bar1, bar2, attempts, successful) in enumerate(zip(bars1, bars2, npca_attempts, npca_successful)):
            if attempts > 0:
                ax_npca.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max_npca_value*0.02,
                            f'{attempts}', ha='center', va='bottom', fontsize=8)
            if successful > 0:
                ax_npca.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max_npca_value*0.02,
                            f'{successful}', ha='center', va='bottom', fontsize=8)
                
                # Add success rate
                if attempts > 0:
                    rate = successful / attempts * 100
                    ax_npca.text(x[i], -max_npca_value*0.1, f'{rate:.1f}%', 
                                ha='center', va='top', fontsize=8, fontweight='bold', color='red')
        
        # Add NPCA status legend
        # if ch_id == 0:  # Only add legend to top row
            # legend_text = "O=OBSS On, X=OBSS Off, N=NPCA On, -=NPCA Off"
            # ax_throughput.text(0.02, 0.98, transform=ax_throughput.transAxes, 
                            #   va='top', ha='left', fontsize=8, 
                            #   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'figure/obss_npca_channel_impact_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_fsm_states_analysis(results):
    """Plot FSM states analysis with separated transmission states"""
    
    config_names = list(results.keys())
    num_configs = len(config_names)
    
    # Create subplot grid based on number of configurations
    cols = min(3, num_configs)
    rows = (num_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Updated state types including separated transmission states
    state_types = ['idle', 'backoff', 'backoff_frozen', 'obss_frozen', 'npca_backoff', 
                   'primary_transmitting', 'npca_transmitting']
    
    # Updated color mapping for states
    state_colors = {
        'idle': '#FFE4E1',              # Light Pink
        'backoff': '#87CEEB',           # Sky Blue
        'backoff_frozen': '#FFA07A',    # Light Salmon
        'obss_frozen': '#FF6347',       # Tomato
        'npca_backoff': '#9370DB',      # Medium Purple
        'primary_transmitting': '#90EE90',  # Light Green
        'npca_transmitting': '#32CD32'      # Lime Green (NEW)
    }
    
    for idx, (config_name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        df = result['dataframe']
        config = result['config']
        stats = result['stats']
        
        ax = axes[idx]
        
        # Count state occurrences for all STAs
        state_counts = {state: 0 for state in state_types}
        
        for ch_id in range(config['num_channels']):
            states_col = f'states_ch_{ch_id}'
            if states_col in df.columns:
                for slot_states in df[states_col]:
                    for state in slot_states:
                        if state in state_counts:
                            state_counts[state] += 1
        
        # Combine transmission states for percentage calculation if needed
        total_primary_tx = state_counts.get('primary_transmitting', 0)
        total_npca_tx = state_counts.get('npca_transmitting', 0)
        total_tx = total_primary_tx + total_npca_tx
        
        # Create pie chart with custom colors
        values = [count for count in state_counts.values() if count > 0]
        labels = []
        colors_pie = []
        
        for state, count in state_counts.items():
            if count > 0:
                if state == 'primary_transmitting':
                    label = f'Primary TX\n({count:,})'
                elif state == 'npca_transmitting':
                    label = f'NPCA TX\n({count:,})'
                else:
                    label = f'{state.replace("_", " ").title()}\n({count:,})'
                
                labels.append(label)
                colors_pie.append(state_colors[state])
        
        if sum(values) > 0:
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90)
            
            # Make percentage text bold and adjust font size
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            # Adjust label font size
            for text in texts:
                text.set_fontsize(8)
        
        # Add configuration info to title
        obss_config = stats['obss_enabled_per_channel']
        npca_config = stats.get('npca_enabled_per_channel', [False] * len(obss_config))
        
        active_obss_channels = [i for i, enabled in enumerate(obss_config) if enabled]
        active_npca_channels = [i for i, enabled in enumerate(npca_config) if enabled]
        
        obss_info = f"OBSS: Ch{active_obss_channels}" if active_obss_channels else "No OBSS"
        npca_info = f"NPCA: Ch{active_npca_channels}" if active_npca_channels else "No NPCA"
        
        # Add STA count info
        total_stas = sum(config['stas_per_channel'])
        stas_info = f"STAs: {config['stas_per_channel']} (Total: {total_stas})"
        
        ax.set_title(f'{config_name}\n{stas_info}\n{obss_info} | {npca_info}\nFSM State Distribution', 
                    fontsize=10, fontweight='bold')
        
        # Add transmission breakdown and NPCA statistics
        info_text = ""
        if total_tx > 0:
            npca_tx_ratio = (total_npca_tx / total_tx * 100) if total_tx > 0 else 0
            info_text += f"TX: {npca_tx_ratio:.1f}% NPCA\n"
        
        if stats.get('npca_total_attempts', 0) > 0:
            info_text += (f"NPCA: {stats['npca_total_successful']}/{stats['npca_total_attempts']} "
                         f"({stats['npca_success_rate']:.1f}%)")
        
        if info_text:
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                   verticalalignment='bottom')
    
    # Remove empty subplots
    for i in range(num_configs, len(axes)):
        axes[i].remove()
    
    # Add overall legend for states
    if num_configs > 0:
        legend_elements = []
        for state, color in state_colors.items():
            if state == 'primary_transmitting':
                label = 'Primary Transmitting'
            elif state == 'npca_transmitting':
                label = 'NPCA Transmitting'
            else:
                label = state.replace("_", " ").title()
            
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        
        # Place legend outside the subplots
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5),
                  title='FSM States', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend
    plt.savefig(f'figure/obss_npca_fsm_states_detailed_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_obss_deferrals_analysis(results):
    """Plot analysis of OBSS deferrals impact and NPCA mitigation with detailed insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: Total Deferrals Breakdown (OBSS vs Intra-BSS)
    ax1 = axes[0, 0]
    
    total_obss_deferrals = []
    total_intra_deferrals = []
    npca_mitigation = []  # NPCA successful transmissions as mitigation
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for sta_stats in stats['stations'].values())
        npca_successful = stats.get('npca_total_successful', 0)
        
        total_obss_deferrals.append(obss_deferrals)
        total_intra_deferrals.append(intra_deferrals)
        npca_mitigation.append(npca_successful)
    
    x = np.arange(len(config_names))
    width = 0.25
    
    bars1 = ax1.bar(x - width, total_obss_deferrals, width, alpha=0.7, 
                    color='lightcoral', label='OBSS Deferrals')
    bars2 = ax1.bar(x, total_intra_deferrals, width, alpha=0.7, 
                    color='lightblue', label='Intra-BSS Deferrals')
    bars3 = ax1.bar(x + width, npca_mitigation, width, alpha=0.7, 
                    color='lightgreen', label='NPCA Successful (Mitigation)')
    
    ax1.set_title('Deferrals vs NPCA Mitigation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels and NPCA effectiveness
    max_value = max(max(total_obss_deferrals), max(total_intra_deferrals), max(npca_mitigation))
    
    for i, (bar1, bar2, bar3, obss_def, npca_succ) in enumerate(zip(bars1, bars2, bars3, total_obss_deferrals, npca_mitigation)):
        # Value labels
        if bar1.get_height() > 0:
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + max_value*0.01,
                    f'{int(bar1.get_height())}', ha='center', va='bottom', fontsize=8)
        if bar2.get_height() > 0:
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + max_value*0.01,
                    f'{int(bar2.get_height())}', ha='center', va='bottom', fontsize=8)
        if bar3.get_height() > 0:
            ax1.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + max_value*0.01,
                    f'{int(bar3.get_height())}', ha='center', va='bottom', fontsize=8)
        
        # NPCA mitigation ratio
        if obss_def > 0 and npca_succ > 0:
            mitigation_ratio = npca_succ / obss_def * 100
            ax1.text(x[i], -max_value*0.08, f'{mitigation_ratio:.1f}%', 
                    ha='center', va='top', fontsize=8, fontweight='bold', color='green')
    
    ax1.text(0.02, 0.02, 'Green % = NPCA Mitigation Ratio vs OBSS Deferrals', 
            transform=ax1.transAxes, va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Plot 2: OBSS Deferrals vs System Performance
    ax2 = axes[0, 1]
    
    throughputs = []
    aoi_values = []
    obss_deferrals_list = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        throughput = (total_successful * config['frame_size']) / stats['total_slots']
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        
        throughputs.append(throughput)
        aoi_values.append(avg_aoi)
        obss_deferrals_list.append(obss_deferrals)
    
    # Create dual-axis plot
    ax2_twin = ax2.twinx()
    
    # Scatter plot: OBSS deferrals vs throughput
    scatter1 = ax2.scatter(obss_deferrals_list, throughputs, s=100, alpha=0.7, 
                          c=colors, marker='o', label='Throughput', edgecolors='black')
    
    # Scatter plot: OBSS deferrals vs AoI (on secondary axis)
    scatter2 = ax2_twin.scatter(obss_deferrals_list, aoi_values, s=100, alpha=0.7, 
                               c=colors, marker='^', label='AoI', edgecolors='red')
    
    # Add configuration labels and NPCA info
    for i, config_name in enumerate(config_names):
        npca_config = results[config_name]['stats'].get('npca_enabled_per_channel', [])
        npca_info = f" +NPCA" if any(npca_config) else ""
        ax2.annotate(f'{config_name}{npca_info}', 
                    (obss_deferrals_list[i], throughputs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('OBSS Deferrals vs Performance Metrics', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total OBSS Deferrals')
    ax2.set_ylabel('System Throughput', color='blue')
    ax2_twin.set_ylabel('Average AoI (μs)', color='red')
    ax2.grid(True, alpha=0.3)
    
    # Legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Channel-specific NPCA Effectiveness
    ax3 = axes[1, 0]
    
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    # Prepare data for each channel
    channel_data = {}
    for ch_id in range(num_channels):
        channel_data[ch_id] = {
            'obss_deferrals': [],
            'npca_attempts': [],
            'npca_successful': [],
            'npca_effectiveness': []
        }
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
            ch_npca_attempts = sum(sta_stats.get('npca_attempts', 0) for _, sta_stats in channel_stas)
            ch_npca_successful = sum(sta_stats.get('npca_successful', 0) for _, sta_stats in channel_stas)
            
            # Calculate NPCA effectiveness (successful NPCA vs OBSS deferrals ratio)
            effectiveness = (ch_npca_successful / ch_obss_deferrals * 100) if ch_obss_deferrals > 0 else 0
            
            channel_data[ch_id]['obss_deferrals'].append(ch_obss_deferrals)
            channel_data[ch_id]['npca_attempts'].append(ch_npca_attempts)
            channel_data[ch_id]['npca_successful'].append(ch_npca_successful)
            channel_data[ch_id]['npca_effectiveness'].append(effectiveness)
    
    # Create grouped bar chart showing NPCA effectiveness per channel
    x = np.arange(len(config_names))
    width = 0.8 / num_channels
    
    for ch_id in range(num_channels):
        offset = (ch_id - num_channels/2 + 0.5) * width
        effectiveness_values = channel_data[ch_id]['npca_effectiveness']
        
        bars = ax3.bar(x + offset, effectiveness_values, width, alpha=0.7, 
                      color=plt.cm.tab10(ch_id), label=f'Channel {ch_id}')
        
        # Add value labels for non-zero values
        for bar, value in zip(bars, effectiveness_values):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax3.set_title('NPCA Effectiveness by Channel', fontsize=12, fontweight='bold')
    ax3.set_ylabel('NPCA Effectiveness (%)\n(Successful NPCA / OBSS Deferrals)')
    ax3.set_xlabel('Configuration')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: NPCA vs Traditional Backoff Analysis
    ax4 = axes[1, 1]
    
    # Compare traditional transmission success vs NPCA success
    traditional_success = []
    npca_success = []
    total_attempts = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        # Traditional successful transmissions (excluding NPCA)
        traditional_succ = 0
        npca_succ = stats.get('npca_total_successful', 0)
        total_att = 0
        
        for sta_stats in stats['stations'].values():
            sta_total_succ = sta_stats['successful_transmissions']
            sta_npca_succ = sta_stats.get('npca_successful', 0)
            sta_traditional_succ = sta_total_succ - sta_npca_succ
            
            traditional_succ += sta_traditional_succ
            total_att += sta_stats['total_attempts']
        
        traditional_success.append(traditional_succ)
        npca_success.append(npca_succ)
        total_attempts.append(total_att)
    
    # Create stacked bar chart
    bars1 = ax4.bar(config_names, traditional_success, alpha=0.7, 
                    color='lightblue', label='Traditional Success')
    bars2 = ax4.bar(config_names, npca_success, bottom=traditional_success, alpha=0.7, 
                    color='lightgreen', label='NPCA Success')
    
    ax4.set_title('Transmission Success: Traditional vs NPCA', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Successful Transmissions Count')
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (config_name, trad, npca, total_att) in enumerate(zip(config_names, traditional_success, npca_success, total_attempts)):
        total_succ = trad + npca
        if total_succ > 0:
            npca_ratio = npca / total_succ * 100
            overall_success_rate = total_succ / total_att * 100 if total_att > 0 else 0
            
            # NPCA contribution
            if npca > 0:
                ax4.text(i, trad + npca/2, f'{npca_ratio:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Overall success rate
            ax4.text(i, total_succ + max(traditional_success)*0.05, f'{overall_success_rate:.1f}%', 
                    ha='center', va='bottom', fontsize=8, color='red')
    
    ax4.text(0.02, 0.98, 'Red % = Overall Success Rate', 
            transform=ax4.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'figure/obss_deferrals_npca_analysis_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_mutual_interference_analysis(results):
    """Plot mutual interference analysis between intra-BSS, OBSS, and NPCA with detailed insights"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: OBSS vs NPCA Traffic Generation
    ax1 = axes[0, 0]
    
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    x = np.arange(len(config_names))
    width = 0.35
    
    # Collect OBSS and NPCA data
    total_obss_generated = []
    total_npca_successful = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        obss_gen = stats.get('obss_events_generated', 0)
        npca_success = stats.get('npca_total_successful', 0)
        
        total_obss_generated.append(obss_gen)
        total_npca_successful.append(npca_success)
    
    bars1 = ax1.bar(x - width/2, total_obss_generated, width, alpha=0.7, 
                    color='lightcoral', label='OBSS Events Generated')
    bars2 = ax1.bar(x + width/2, total_npca_successful, width, alpha=0.7, 
                    color='lightgreen', label='NPCA Successful Transmissions')
    
    ax1.set_title('OBSS vs NPCA Traffic Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: NPCA Impact on Channel Utilization
    ax2 = axes[0, 1]
    
    # Show how NPCA affects different channels
    channel_impact_data = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        impact_row = []
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                # Calculate channel utilization
                ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                ch_utilization = (ch_successful * config['frame_size']) / stats['total_slots']
                
                # Normalize by number of STAs in channel
                normalized_util = ch_utilization / len(channel_stas) if len(channel_stas) > 0 else 0
                impact_row.append(normalized_util)
            else:
                impact_row.append(0)
        
        channel_impact_data.append(impact_row)
    
    # Create heatmap
    im = ax2.imshow(channel_impact_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    ax2.set_xticks(range(num_channels))
    ax2.set_xticklabels([f'Channel {i}' for i in range(num_channels)])
    ax2.set_yticks(range(len(config_names)))
    ax2.set_yticklabels(config_names, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Normalized Channel Utilization', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(config_names)):
        for j in range(num_channels):
            value = channel_impact_data[i][j]
            text_color = 'white' if value > np.max(channel_impact_data) * 0.5 else 'black'
            ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=10)
    
    ax2.set_title('NPCA Impact on Channel Utilization', fontsize=12, fontweight='bold')
    
    # Plot 3: NPCA Efficiency Analysis
    ax3 = axes[1, 0]
    
    # Analyze NPCA efficiency vs STA density
    sta_counts = []
    npca_success_rates = []
    npca_utilization_rates = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # Count STAs in NPCA-enabled channels
        npca_stas = 0
        for ch_id, enabled in enumerate(stats.get('npca_enabled_per_channel', [])):
            if enabled:
                npca_stas += config['stas_per_channel'][ch_id]
        
        sta_counts.append(npca_stas)
        npca_success_rates.append(stats.get('npca_success_rate', 0))
        npca_utilization_rates.append(stats.get('npca_utilization_rate', 0))
    
    # Create scatter plot with different sizes based on utilization
    scatter = ax3.scatter(sta_counts, npca_success_rates, 
                         s=[rate*50 for rate in npca_utilization_rates], 
                         alpha=0.7, c=colors, edgecolors='black', linewidth=1)
    
    for i, config_name in enumerate(config_names):
        ax3.annotate(config_name, (sta_counts[i], npca_success_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_title('NPCA Efficiency vs STA Density', fontsize=12, fontweight='bold')
    ax3.set_xlabel('NPCA-enabled STAs Count')
    ax3.set_ylabel('NPCA Success Rate (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add size legend
    ax3.text(0.02, 0.98, 'Bubble size = NPCA Utilization Rate', 
            transform=ax3.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 4: Cross-Channel NPCA Traffic Analysis
    ax4 = axes[1, 1]
    
    # Analyze NPCA traffic flow between channels
    npca_flow_data = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        # For each source channel, count NPCA attempts and successes
        flow_row = []
        for source_ch in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == source_ch]
            
            total_npca_attempts = sum(sta_stats.get('npca_attempts', 0) for _, sta_stats in channel_stas)
            total_npca_successful = sum(sta_stats.get('npca_successful', 0) for _, sta_stats in channel_stas)
            
            # Calculate NPCA activity level
            npca_activity = total_npca_successful if total_npca_successful > 0 else 0
            flow_row.append(npca_activity)
        
        npca_flow_data.append(flow_row)
    
    # Create stacked bar chart showing NPCA flow
    bottom = np.zeros(len(config_names))
    colors_flow = plt.cm.tab10(np.linspace(0, 1, num_channels))
    
    for ch_id in range(num_channels):
        values = [row[ch_id] for row in npca_flow_data]
        bars = ax4.bar(config_names, values, bottom=bottom, alpha=0.7, 
                      color=colors_flow[ch_id], label=f'From Channel {ch_id}')
        
        # Add value labels for non-zero values
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bottom[i] + value/2,
                        f'{int(value)}', ha='center', va='center', 
                        fontweight='bold', fontsize=9)
        
        bottom += values
    
    ax4.set_title('NPCA Cross-Channel Traffic Flow', fontsize=12, fontweight='bold')
    ax4.set_ylabel('NPCA Successful Transmissions')
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add configuration info
    for i, config_name in enumerate(config_names):
        npca_config = results[config_name]['stats'].get('npca_enabled_per_channel', [])
        active_npca = [j for j, enabled in enumerate(npca_config) if enabled]
        if active_npca:
            ax4.text(i, -max([sum(row) for row in npca_flow_data])*0.1, 
                    f"NPCA:Ch{active_npca}", ha='center', va='top', 
                    fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'figure/mutual_interference_npca_analysis_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_channel_fairness_analysis(results):
    """Plot channel access fairness analysis with NPCA impact considerations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    num_channels = len(results[config_names[0]]['stats']['obss_enabled_per_channel'])
    
    # Plot 1: Per-Channel Utilization with OBSS/NPCA Status
    ax1 = axes[0, 0]
    
    for idx, (config_name, result) in enumerate(results.items()):
        stats = result['stats']
        config = result['config']
        obss_config = stats['obss_enabled_per_channel']
        npca_config = stats.get('npca_enabled_per_channel', [False] * num_channels)
        
        channel_utils = []
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            total_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
            util = (total_successful * config['frame_size']) / stats['total_slots']
            channel_utils.append(util)
        
        x_pos = np.arange(len(channel_utils)) + idx * 0.15
        bars = ax1.bar(x_pos, channel_utils, width=0.12, alpha=0.7, 
                      color=colors[idx], label=config_name)
        
        # Add value labels and OBSS/NPCA status
        for i, (util, bar) in enumerate(zip(channel_utils, bars)):
            ax1.text(bar.get_x() + bar.get_width()/2, util + max(channel_utils)*0.02, 
                    f'{util:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add status indicators with symbols
            status_symbols = ""
            if obss_config[i]:
                status_symbols += "O"  # OBSS enabled
            else:
                status_symbols += "."   # OBSS disabled
            
            if npca_config[i]:
                status_symbols += "N"  # NPCA enabled
            else:
                status_symbols += "."   # NPCA disabled
            
            # Color coding: red for OBSS, blue for NPCA
            obss_color = 'red' if obss_config[i] else 'lightgray'
            npca_color = 'blue' if npca_config[i] else 'lightgray'
            
            ax1.text(bar.get_x() + bar.get_width()/2, util + max(channel_utils)*0.08, 
                    status_symbols[0], ha='center', va='bottom', fontsize=10, 
                    color=obss_color, fontweight='bold')
            ax1.text(bar.get_x() + bar.get_width()/2, util + max(channel_utils)*0.12, 
                    status_symbols[1], ha='center', va='bottom', fontsize=10, 
                    color=npca_color, fontweight='bold')
    
    ax1.set_title('Channel Utilization with OBSS/NPCA Status', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Channel Utilization')
    ax1.set_xlabel('Channel ID')
    ax1.set_xticks(np.arange(num_channels) + 0.3)
    ax1.set_xticklabels([f'Channel {i}' for i in range(num_channels)])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for status indicators
    ax1.text(0.02, 0.98, 'O=OBSS, N=NPCA, .=Disabled', transform=ax1.transAxes, 
            va='top', ha='left', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: System Fairness with NPCA Impact
    ax2 = axes[0, 1]
    
    overall_fairness = []
    intra_channel_fairness = []  # Average fairness within channels
    inter_channel_fairness = []  # Fairness between channels
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # 1. Overall fairness (all STAs)
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        if sta_throughputs:
            sum_throughputs = sum(sta_throughputs)
            sum_squared = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_throughputs**2) / (n * sum_squared) if sum_squared > 0 else 0
        else:
            fairness = 0
        overall_fairness.append(fairness)
        
        # 2. Intra-channel fairness (average across channels)
        channel_fairness_values = []
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if len(channel_stas) > 1:
                ch_throughputs = [(sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                 for _, sta_stats in channel_stas]
                ch_sum = sum(ch_throughputs)
                ch_sum_sq = sum(t**2 for t in ch_throughputs)
                ch_n = len(ch_throughputs)
                ch_fairness = (ch_sum**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
            else:
                ch_fairness = 1.0  # Perfect fairness for single STA
            
            channel_fairness_values.append(ch_fairness)
        
        intra_channel_fairness.append(np.mean(channel_fairness_values))
        
        # 3. Inter-channel fairness
        channel_avg_throughputs = []
        for ch_id in range(num_channels):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                           if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                ch_total_tp = sum((sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                for _, sta_stats in channel_stas)
                ch_avg_tp = ch_total_tp / len(channel_stas)  # Per-STA average
                channel_avg_throughputs.append(ch_avg_tp)
        
        if len(channel_avg_throughputs) > 1:
            ch_sum = sum(channel_avg_throughputs)
            ch_sum_sq = sum(t**2 for t in channel_avg_throughputs)
            ch_n = len(channel_avg_throughputs)
            inter_fairness = (ch_sum**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
        else:
            inter_fairness = 1.0
        
        inter_channel_fairness.append(inter_fairness)
    
    # Plot fairness metrics as grouped bars
    x = np.arange(len(config_names))
    width = 0.25
    
    bars1 = ax2.bar(x - width, overall_fairness, width, alpha=0.7, 
                    color='skyblue', label='Overall Fairness')
    bars2 = ax2.bar(x, intra_channel_fairness, width, alpha=0.7, 
                    color='lightgreen', label='Avg Intra-Channel Fairness')
    bars3 = ax2.bar(x + width, inter_channel_fairness, width, alpha=0.7, 
                    color='lightcoral', label='Inter-Channel Fairness')
    
    ax2.set_title('Multi-level Fairness Analysis', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Jain\'s Fairness Index')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: NPCA Impact on Per-STA Performance
    ax3 = axes[1, 0]
    
    # Analyze per-STA performance based on NPCA usage
    npca_users_perf = []
    non_npca_users_perf = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        npca_perf = []
        non_npca_perf = []
        
        for sta_id, sta_stats in stats['stations'].items():
            throughput = (sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots']
            
            if sta_stats.get('npca_enabled', False):
                npca_perf.append(throughput)
            else:
                non_npca_perf.append(throughput)
        
        npca_users_perf.append(npca_perf)
        non_npca_users_perf.append(non_npca_perf)
    
    # Create box plots for each configuration
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, config_name in enumerate(config_names):
        if non_npca_users_perf[i]:
            box_data.append(non_npca_users_perf[i])
            box_labels.append(f'{config_name}\n(Non-NPCA)')
            box_colors.append('lightblue')
        
        if npca_users_perf[i]:
            box_data.append(npca_users_perf[i])
            box_labels.append(f'{config_name}\n(NPCA)')
            box_colors.append('lightgreen')
    
    if box_data:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax3.set_title('Per-STA Performance: NPCA vs Non-NPCA', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput per STA')
    ax3.set_xticklabels(box_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: NPCA Fairness Contribution Analysis
    ax4 = axes[1, 1]
    
    # Analyze how NPCA affects fairness in different scenarios
    sta_densities = []
    fairness_improvements = []
    npca_contributions = []
    
    # Find baseline (no NPCA) configuration
    baseline_fairness = None
    for config_name in config_names:
        npca_config = results[config_name]['stats'].get('npca_enabled_per_channel', [])
        if not any(npca_config):  # No NPCA
            baseline_fairness = overall_fairness[config_names.index(config_name)]
            break
    
    for i, config_name in enumerate(config_names):
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # Count STAs in NPCA-enabled channels
        npca_stas = 0
        total_stas = sum(config['stas_per_channel'])
        npca_config = stats.get('npca_enabled_per_channel', [])
        
        for ch_id, enabled in enumerate(npca_config):
            if enabled:
                npca_stas += config['stas_per_channel'][ch_id]
        
        sta_densities.append(total_stas)
        
        # Calculate fairness improvement
        if baseline_fairness is not None:
            improvement = overall_fairness[i] - baseline_fairness
            fairness_improvements.append(improvement)
        else:
            fairness_improvements.append(0)
        
        # NPCA contribution to total throughput
        npca_contribution = stats.get('npca_total_successful', 0)
        total_transmissions = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        npca_ratio = (npca_contribution / total_transmissions * 100) if total_transmissions > 0 else 0
        npca_contributions.append(npca_ratio)
    
    # Create scatter plot
    scatter = ax4.scatter(npca_contributions, fairness_improvements, 
                         s=[density*5 for density in sta_densities], 
                         alpha=0.7, c=colors, edgecolors='black', linewidth=1)
    
    for i, config_name in enumerate(config_names):
        ax4.annotate(config_name, (npca_contributions[i], fairness_improvements[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Improvement')
    ax4.set_title('NPCA Contribution vs Fairness Improvement', fontsize=12, fontweight='bold')
    ax4.set_xlabel('NPCA Contribution to Total Throughput (%)')
    ax4.set_ylabel('Fairness Improvement vs Baseline')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add size legend
    ax4.text(0.02, 0.98, 'Bubble size = Total STAs', 
            transform=ax4.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'figure/channel_fairness_npca_analysis_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_npca_cross_channel_impact(results):
    """Plot analysis of how Channel 1 NPCA usage affects Channel 0 performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(config_names)))
    
    # Plot 1: Channel 0 Performance Impact
    ax1 = axes[0, 0]
    
    # Collect Channel 0 performance metrics
    ch0_throughputs = []
    ch0_aois = []
    ch0_collisions = []
    npca_loads_on_ch0 = []  # NPCA traffic received by Channel 0
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # Channel 0 STAs performance
        ch0_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                   if sta_stats['channel'] == 0]
        
        if ch0_stas:
            ch0_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in ch0_stas)
            ch0_throughput = (ch0_successful * config['frame_size']) / stats['total_slots']
            ch0_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in ch0_stas])
            ch0_collision = sum(sta_stats['collisions'] for _, sta_stats in ch0_stas)
            
            ch0_throughputs.append(ch0_throughput)
            ch0_aois.append(ch0_aoi)
            ch0_collisions.append(ch0_collision)
        else:
            ch0_throughputs.append(0)
            ch0_aois.append(0)
            ch0_collisions.append(0)
        
        # Estimate NPCA load on Channel 0 (from Channel 1 STAs)
        ch1_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                   if sta_stats['channel'] == 1]
        npca_load = sum(sta_stats.get('npca_successful', 0) for _, sta_stats in ch1_stas)
        npca_loads_on_ch0.append(npca_load)
    
    # Create grouped bar chart for Channel 0 metrics
    x = np.arange(len(config_names))
    width = 0.25
    
    # Normalize metrics for comparison
    max_throughput = max(ch0_throughputs) if ch0_throughputs else 1
    max_aoi = max(ch0_aois) if ch0_aois else 1
    max_collision = max(ch0_collisions) if ch0_collisions else 1
    
    normalized_throughput = [t/max_throughput for t in ch0_throughputs]
    normalized_aoi = [a/max_aoi for a in ch0_aois]
    normalized_collision = [c/max_collision for c in ch0_collisions]
    
    bars1 = ax1.bar(x - width, normalized_throughput, width, alpha=0.7, 
                    color='skyblue', label='Throughput (normalized)')
    bars2 = ax1.bar(x, normalized_aoi, width, alpha=0.7, 
                    color='lightcoral', label='AoI (normalized)')
    bars3 = ax1.bar(x + width, normalized_collision, width, alpha=0.7, 
                    color='orange', label='Collisions (normalized)')
    
    ax1.set_title('Channel 0 Performance Impact', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add NPCA load indicators
    for i, npca_load in enumerate(npca_loads_on_ch0):
        if npca_load > 0:
            ax1.text(i, 1.1, f'NPCA:{npca_load}', ha='center', va='bottom', 
                    fontsize=8, color='red', fontweight='bold')
    
    # Add actual values as text
    ax1.text(0.02, 0.98, f'Max: TP={max_throughput:.4f}, AoI={max_aoi:.0f}μs, Col={max_collision}', 
            transform=ax1.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Plot 2: NPCA Load vs Channel 0 Performance Degradation
    ax2 = axes[0, 1]
    
    # Calculate performance degradation relative to baseline (first config)
    baseline_throughput = ch0_throughputs[0] if ch0_throughputs else 0
    baseline_aoi = ch0_aois[0] if ch0_aois else 0
    
    throughput_degradation = []
    aoi_degradation = []
    
    for i, (tp, aoi) in enumerate(zip(ch0_throughputs, ch0_aois)):
        tp_deg = ((baseline_throughput - tp) / baseline_throughput * 100) if baseline_throughput > 0 else 0
        aoi_deg = ((aoi - baseline_aoi) / baseline_aoi * 100) if baseline_aoi > 0 else 0
        
        throughput_degradation.append(tp_deg)
        aoi_degradation.append(aoi_deg)
    
    # Scatter plot: NPCA load vs performance degradation
    scatter1 = ax2.scatter(npca_loads_on_ch0, throughput_degradation, s=100, alpha=0.7, 
                          c=colors, marker='o', label='Throughput Degradation (%)', edgecolors='blue')
    scatter2 = ax2.scatter(npca_loads_on_ch0, aoi_degradation, s=100, alpha=0.7, 
                          c=colors, marker='^', label='AoI Degradation (%)', edgecolors='red')
    
    # Add configuration labels
    for i, config_name in enumerate(config_names):
        ax2.annotate(config_name, (npca_loads_on_ch0[i], throughput_degradation[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Degradation')
    ax2.set_title('NPCA Load vs Channel 0 Performance Degradation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('NPCA Load on Channel 0 (successful transmissions)')
    ax2.set_ylabel('Performance Degradation (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Channel Utilization Balance Analysis
    ax3 = axes[1, 0]
    
    # Compare Channel 0 vs Channel 1 utilization
    ch1_throughputs = []
    utilization_ratios = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        config = results[config_name]['config']
        
        # Channel 1 STAs performance
        ch1_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                   if sta_stats['channel'] == 1]
        
        if ch1_stas:
            ch1_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in ch1_stas)
            ch1_throughput = (ch1_successful * config['frame_size']) / stats['total_slots']
            ch1_throughputs.append(ch1_throughput)
        else:
            ch1_throughputs.append(0)
        
        # Calculate utilization ratio (Ch1/Ch0)
        if ch0_throughputs[config_names.index(config_name)] > 0:
            ratio = ch1_throughputs[-1] / ch0_throughputs[config_names.index(config_name)]
            utilization_ratios.append(ratio)
        else:
            utilization_ratios.append(0)
    
    # Create dual bar chart
    x = np.arange(len(config_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, ch0_throughputs, width, alpha=0.7, 
                    color='lightblue', label='Channel 0 Throughput')
    bars2 = ax3.bar(x + width/2, ch1_throughputs, width, alpha=0.7, 
                    color='lightgreen', label='Channel 1 Throughput')
    
    ax3.set_title('Channel Utilization Balance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add ratio labels
    for i, (bar1, bar2, ratio) in enumerate(zip(bars1, bars2, utilization_ratios)):
        max_height = max(bar1.get_height(), bar2.get_height())
        ax3.text(i, max_height + max(max(ch0_throughputs), max(ch1_throughputs))*0.05, 
                f'Ratio: {ratio:.2f}', ha='center', va='bottom', fontsize=8, 
                color='purple', fontweight='bold')
    
    # Add fairness line
    ideal_ratio = 1.0
    ax3.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Perfect Balance (1:1)')
    
    # Plot 4: NPCA Efficiency vs Channel 0 Impact
    ax4 = axes[1, 1]
    
    # Calculate NPCA efficiency metrics
    npca_success_rates = []
    ch0_impact_scores = []
    
    for config_name in config_names:
        stats = results[config_name]['stats']
        
        # NPCA success rate
        npca_attempts = stats.get('npca_total_attempts', 0)
        npca_successful = stats.get('npca_total_successful', 0)
        success_rate = (npca_successful / npca_attempts * 100) if npca_attempts > 0 else 0
        npca_success_rates.append(success_rate)
        
        # Channel 0 impact score (weighted combination of throughput and AoI degradation)
        i = config_names.index(config_name)
        impact_score = (throughput_degradation[i] * 0.5 + aoi_degradation[i] * 0.5)
        ch0_impact_scores.append(impact_score)
    
    # Create bubble chart
    bubble_sizes = [npca_loads_on_ch0[i]*5 for i in range(len(config_names))]
    
    scatter = ax4.scatter(npca_success_rates, ch0_impact_scores, s=bubble_sizes, 
                         alpha=0.7, c=colors, edgecolors='black', linewidth=1)
    
    # Add configuration labels
    for i, config_name in enumerate(config_names):
        ax4.annotate(config_name, (npca_success_rates[i], ch0_impact_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Impact')
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('NPCA Efficiency vs Channel 0 Impact', fontsize=12, fontweight='bold')
    ax4.set_xlabel('NPCA Success Rate (%)')
    ax4.set_ylabel('Channel 0 Impact Score (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax4.text(0.95, 0.95, 'High NPCA Success\nHigh Ch0 Impact', transform=ax4.transAxes, 
            ha='right', va='top', fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax4.text(0.95, 0.05, 'High NPCA Success\nLow Ch0 Impact', transform=ax4.transAxes, 
            ha='right', va='bottom', fontsize=8, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Bubble size legend
    ax4.text(0.02, 0.98, 'Bubble size = NPCA Load on Ch0', 
            transform=ax4.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'figure/npca_cross_channel_impact_{config["label"]}.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Print function was already provided in the previous response, so keeping it as is
def print_obss_results(results):
    """Print detailed OBSS and NPCA comparison results"""
    
    print("\n" + "="*80)
    print("CHANNEL-SPECIFIC OBSS & NPCA INTERFERENCE ANALYSIS")
    print("="*80)
    
    for config_name, result in results.items():
        stats = result['stats']
        config = result['config']
        
        print(f"\n📊 {config_name}")
        print("-" * 50)
        print(f"OBSS Configuration: {stats['obss_enabled_per_channel']}")
        print(f"NPCA Configuration: {stats['npca_enabled_per_channel']}")
        
        # 채널별 OBSS 상태 출력 (기존)
        print(f"\n🔧 Channel-specific OBSS Status:")
        total_obss_channels = sum(stats['obss_enabled_per_channel'])
        print(f"Active OBSS channels: {total_obss_channels}/{config['num_channels']}")
        
        for ch_id in range(config['num_channels']):
            obss_ch_stats = stats['obss_per_channel'][ch_id]
            if obss_ch_stats['enabled']:
                print(f"  Channel {ch_id}: OBSS ENABLED")
                print(f"    Generated: {obss_ch_stats['generated']}")
                print(f"    Deferred: {obss_ch_stats['deferred']}")
                print(f"    Blocked by intra-BSS: {obss_ch_stats['blocked_by_intra']}")
                print(f"    Blocked by other OBSS: {obss_ch_stats['blocked_by_other_obss']}")
                
                total_attempts = obss_ch_stats['generated'] + obss_ch_stats['deferred']
                success_rate = (obss_ch_stats['generated'] / total_attempts * 100) if total_attempts > 0 else 0
                print(f"    Success rate: {success_rate:.1f}%")
            else:
                print(f"  Channel {ch_id}: OBSS DISABLED")
        
        # 채널별 NPCA 상태 출력 (새로 추가)
        print(f"\n🚀 Channel-specific NPCA Status:")
        total_npca_channels = sum(stats['npca_enabled_per_channel'])
        print(f"Active NPCA channels: {total_npca_channels}/{config['num_channels']}")
        
        for ch_id in range(config['num_channels']):
            npca_ch_stats = stats['npca_per_channel'][ch_id]
            if npca_ch_stats['enabled']:
                print(f"  Channel {ch_id}: NPCA ENABLED ({npca_ch_stats['npca_stas_count']} STAs)")
                print(f"    Total attempts: {npca_ch_stats['total_attempts']}")
                print(f"    Successful: {npca_ch_stats['total_successful']}")
                print(f"    Blocked: {npca_ch_stats['total_blocked']}")
                print(f"    Success rate: {npca_ch_stats['success_rate']:.1f}%")
            else:
                print(f"  Channel {ch_id}: NPCA DISABLED")
        
        # 전체 OBSS 통계 (기존)
        if stats['obss_enabled']:
            print(f"\n🌐 Overall OBSS Statistics:")
            print(f"OBSS Generation Rate: {stats['obss_generation_rate']:.1%} per slot")
            print(f"Total OBSS Events Generated: {stats['obss_events_generated']}")
            print(f"Total OBSS Events Deferred: {stats['obss_events_deferred']}")
            print(f"Total OBSS Duration: {stats['obss_total_duration_slots']} slots ({stats['obss_total_duration_us']/1000:.1f} ms)")
            print(f"OBSS Channel Utilization: {stats['obss_channel_utilization']:.1%}")
            
            print(f"\n🔄 Mutual Interference:")
            print(f"  OBSS blocked by Intra-BSS: {stats['obss_blocked_by_intra_bss']}")
            print(f"  OBSS blocked by other OBSS: {stats['obss_blocked_by_other_obss']}")
            print(f"  Total mutual interference events: {stats['mutual_interference_events']}")
            
            total_obss_attempts = stats['obss_events_generated'] + stats['obss_events_deferred']
            obss_success_rate = (stats['obss_events_generated'] / total_obss_attempts * 100) if total_obss_attempts > 0 else 0
            print(f"  Overall OBSS success rate: {obss_success_rate:.1f}%")
        else:
            print(f"\n🌐 Overall OBSS Statistics: DISABLED")
        
        # 전체 NPCA 통계 (새로 추가)
        if stats['npca_enabled_stas'] > 0:
            print(f"\n🚀 Overall NPCA Statistics:")
            print(f"NPCA Enabled STAs: {stats['npca_enabled_stas']}")
            print(f"Total NPCA Attempts: {stats['npca_total_attempts']}")
            print(f"Total NPCA Successful: {stats['npca_total_successful']}")
            print(f"Total NPCA Blocked: {stats['npca_total_blocked']}")
            print(f"NPCA Success Rate: {stats['npca_success_rate']:.1f}%")
            print(f"NPCA Utilization Rate: {stats['npca_utilization_rate']:.3f}% (attempts per slot)")
            
            # NPCA 효과 분석
            if stats['npca_total_successful'] > 0:
                npca_contribution = stats['npca_total_successful'] / sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values()) * 100
                print(f"NPCA Contribution to Total Throughput: {npca_contribution:.1f}%")
        else:
            print(f"\n🚀 Overall NPCA Statistics: DISABLED")
        
        # 시뮬레이션 기본 정보 (기존)
        print(f"\n⏱️ Simulation Parameters:")
        print(f"Simulation time: {stats['total_time_us']/1000:.1f} ms ({stats['total_slots']} slots)")
        print(f"Channels: {config['num_channels']}, STAs per channel: {config['stas_per_channel']}")
        print(f"Frame size: {config['frame_size']} slots")
        
        # 시스템 성능 계산 (기존)
        total_successful = sum(sta_stats['successful_transmissions'] for sta_stats in stats['stations'].values())
        total_attempts = sum(sta_stats['total_attempts'] for sta_stats in stats['stations'].values())
        total_collisions = sum(sta_stats['collisions'] for sta_stats in stats['stations'].values())
        total_obss_deferrals = sum(sta_stats['obss_deferrals'] for sta_stats in stats['stations'].values())
        total_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for sta_stats in stats['stations'].values())
        system_throughput = (total_successful * config['frame_size']) / stats['total_slots']
        system_success_rate = total_successful / total_attempts if total_attempts > 0 else 0
        avg_aoi = np.mean([sta_stats['average_aoi_time_us'] for sta_stats in stats['stations'].values()])
        
        print(f"\n📈 System Performance:")
        print(f"System throughput: {system_throughput:.4f} ({system_throughput*100:.2f}%)")
        print(f"System success rate: {system_success_rate:.4f} ({system_success_rate*100:.2f}%)")
        print(f"Average AoI: {avg_aoi:.1f} μs")
        print(f"Total successful transmissions: {total_successful}")
        print(f"Total collisions: {total_collisions}")
        print(f"Total STA OBSS deferrals: {total_obss_deferrals}")
        print(f"Total STA Intra-BSS deferrals: {total_intra_deferrals}")
        
        # 공정성 지수 계산 (기존)
        sta_throughputs = []
        for sta_stats in stats['stations'].values():
            throughput = (sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots']
            sta_throughputs.append(throughput)
        
        if sta_throughputs:
            sum_tp = sum(sta_throughputs)
            sum_sq = sum(t**2 for t in sta_throughputs)
            n = len(sta_throughputs)
            fairness = (sum_tp**2) / (n * sum_sq) if sum_sq > 0 else 0
            print(f"Jain's Fairness Index: {fairness:.3f}")
        
        # 채널별 상세 분석 (NPCA 정보 포함)
        print(f"\n🔧 Per-Channel Performance Analysis:")
        for ch_id in range(config['num_channels']):
            channel_stas = [(sta_id, sta_stats) for sta_id, sta_stats in stats['stations'].items() 
                          if sta_stats['channel'] == ch_id]
            
            if channel_stas:
                # 채널별 성능 지표 계산
                ch_successful = sum(sta_stats['successful_transmissions'] for _, sta_stats in channel_stas)
                ch_attempts = sum(sta_stats['total_attempts'] for _, sta_stats in channel_stas)
                ch_throughput = (ch_successful * config['frame_size']) / stats['total_slots']
                ch_success_rate = ch_successful / ch_attempts if ch_attempts > 0 else 0
                ch_aoi = np.mean([sta_stats['average_aoi_time_us'] for _, sta_stats in channel_stas])
                ch_collisions = sum(sta_stats['collisions'] for _, sta_stats in channel_stas)
                ch_obss_deferrals = sum(sta_stats['obss_deferrals'] for _, sta_stats in channel_stas)
                ch_intra_deferrals = sum(sta_stats['intra_bss_deferrals'] for _, sta_stats in channel_stas)
                
                # NPCA 통계 계산
                ch_npca_attempts = sum(sta_stats['npca_attempts'] for _, sta_stats in channel_stas)
                ch_npca_successful = sum(sta_stats['npca_successful'] for _, sta_stats in channel_stas)
                ch_npca_blocked = sum(sta_stats['npca_blocked'] for _, sta_stats in channel_stas)
                
                # 상태 표시
                obss_status = "🟢 ENABLED" if stats['obss_enabled_per_channel'][ch_id] else "🔴 DISABLED"
                npca_status = "🚀 ENABLED" if stats['npca_enabled_per_channel'][ch_id] else "⚫ DISABLED"
                
                print(f"  Channel {ch_id} ({len(channel_stas)} STAs) - OBSS {obss_status}, NPCA {npca_status}:")
                print(f"    Throughput: {ch_throughput:.4f} ({ch_throughput*100:.2f}%)")
                print(f"    Success rate: {ch_success_rate:.3f} ({ch_success_rate*100:.1f}%)")
                print(f"    Average AoI: {ch_aoi:.1f} μs")
                print(f"    Collisions: {ch_collisions}")
                print(f"    OBSS deferrals: {ch_obss_deferrals}")
                print(f"    Intra-BSS deferrals: {ch_intra_deferrals}")
                print(f"    NPCA attempts: {ch_npca_attempts}")
                print(f"    NPCA successful: {ch_npca_successful}")
                print(f"    NPCA blocked: {ch_npca_blocked}")
                if ch_npca_attempts > 0:
                    print(f"    NPCA success rate: {ch_npca_successful / ch_npca_attempts * 100:.1f}%")
                
                # 채널 내 공정성 (기존)
                ch_sta_throughputs = [(sta_stats['successful_transmissions'] * config['frame_size']) / stats['total_slots'] 
                                     for _, sta_stats in channel_stas]
                if len(ch_sta_throughputs) > 1:
                    ch_sum_tp = sum(ch_sta_throughputs)
                    ch_sum_sq = sum(t**2 for t in ch_sta_throughputs)
                    ch_n = len(ch_sta_throughputs)
                    ch_fairness = (ch_sum_tp**2) / (ch_n * ch_sum_sq) if ch_sum_sq > 0 else 0
                    print(f"    Channel fairness: {ch_fairness:.3f}")
    
    print(f"\n✅ NPCA Analysis Complete!")
    print(f"📋 Summary: Analyzed {len(results)} configurations with NPCA support")