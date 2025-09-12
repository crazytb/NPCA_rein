#!/usr/bin/env python3
"""
Extract the DRL Action Distribution plot from comparison results
Save as separate PNG and EPS files
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_reward_distribution_plot():
    """Create and save the DRL Action Distribution plot separately"""
    
    # Load the comparison data
    df = pd.read_csv('./comparison_results/new_reward_comparison.csv')
    
    # Filter DRL data only
    drl_data = df[df['policy'].str.contains('DRL')].copy()
    
    if drl_data.empty:
        print("No DRL data found in comparison results")
        return
    
    # Sort by OBSS duration for proper ordering
    drl_data = drl_data.sort_values('obss_duration')
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    x_positions = range(len(drl_data))
    bars = plt.bar(x_positions, drl_data['action_stay_ratio'], 
                   color='lightblue', edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Customize the plot
    plt.xlabel('DRL Model (by OBSS Duration)', fontsize=14, fontweight='bold')
    plt.ylabel('Stay Primary Ratio', fontsize=14, fontweight='bold')
    plt.title('DRL Action Distribution', fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(x_positions, drl_data['policy'], rotation=0, fontsize=12)
    
    # Set y-axis to percentage format
    plt.ylim(0, 1.0)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, drl_data['action_stay_ratio'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig('./comparison_results/reward_distribution.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 Reward distribution plot saved as PNG: ./comparison_results/reward_distribution.png")
    
    # Save as EPS
    plt.savefig('./comparison_results/reward_distribution.eps', 
                format='eps', bbox_inches='tight', facecolor='white')
    print("📊 Reward distribution plot saved as EPS: ./comparison_results/reward_distribution.eps")
    
    # Show additional statistics
    print(f"\n📈 DRL Action Statistics:")
    for _, row in drl_data.iterrows():
        stay_pct = row['action_stay_ratio'] * 100
        npca_pct = (1 - row['action_stay_ratio']) * 100
        print(f"  {row['policy']}: {stay_pct:.1f}% Stay Primary, {npca_pct:.1f}% Go NPCA")
        print(f"    Avg Reward: {row['avg_reward']:.2f}, Throughput: {row['avg_throughput']:.1f} slots")
    
    plt.show()

if __name__ == "__main__":
    create_reward_distribution_plot()