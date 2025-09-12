#!/usr/bin/env python3
"""
Extract the reward distribution subplot from policy_comparison.png
Save as separate PNG and EPS files
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

def extract_reward_distribution_subplot():
    """Extract the bottom-left subplot from policy_comparison.png"""
    
    # Read the original image
    img = mpimg.imread('./comparison_results/policy_comparison.png')
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate subplot boundaries (bottom-left quadrant)
    # Assuming 2x2 subplot layout with some margin
    left_margin = int(width * 0.08)    # Left margin
    right_boundary = int(width * 0.52)  # Middle of image
    bottom_boundary = int(height * 0.48) # Middle of image  
    bottom_margin = int(height * 0.05)   # Bottom margin
    
    # Extract the bottom-left subplot
    subplot_img = img[bottom_boundary:height-bottom_margin, left_margin:right_boundary]
    
    # Create new figure with the extracted subplot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(subplot_img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save as PNG
    plt.savefig('./comparison_results/reward_distribution_subplot.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    print("📊 Reward distribution subplot saved as PNG: ./comparison_results/reward_distribution_subplot.png")
    
    # Save as EPS
    plt.savefig('./comparison_results/reward_distribution_subplot.eps', 
                format='eps', bbox_inches='tight', facecolor='white', pad_inches=0)
    print("📊 Reward distribution subplot saved as EPS: ./comparison_results/reward_distribution_subplot.eps")
    
    plt.close()

def recreate_reward_distribution_from_data():
    """Recreate the reward distribution plot from the CSV data"""
    
    try:
        import pandas as pd
        
        # Load comparison data
        df = pd.read_csv('./comparison_results/new_reward_comparison.csv')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter data for plotting
        policies = df['policy'].unique()
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightpink']
        
        # Group by policy and get average reward
        policy_rewards = df.groupby('policy')['avg_reward'].mean()
        
        # Create bar plot
        bars = ax.bar(range(len(policy_rewards)), policy_rewards.values, 
                     color=colors[:len(policy_rewards)], 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Policy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
        ax.set_title('Policy Reward Distribution', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(policy_rewards)))
        ax.set_xticklabels(policy_rewards.index, rotation=45, ha='right', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, policy_rewards.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save as PNG
        plt.savefig('./comparison_results/policy_reward_distribution.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        print("📊 Policy reward distribution plot saved as PNG: ./comparison_results/policy_reward_distribution.png")
        
        # Save as EPS
        plt.savefig('./comparison_results/policy_reward_distribution.eps', 
                    format='eps', bbox_inches='tight', facecolor='white')
        print("📊 Policy reward distribution plot saved as EPS: ./comparison_results/policy_reward_distribution.eps")
        
        plt.close()
        
    except Exception as e:
        print(f"Could not recreate from data: {e}")

if __name__ == "__main__":
    print("🎯 Extracting reward distribution subplot from policy_comparison.png")
    extract_reward_distribution_subplot()
    
    print("\n🎯 Also creating clean version from data")
    recreate_reward_distribution_from_data()