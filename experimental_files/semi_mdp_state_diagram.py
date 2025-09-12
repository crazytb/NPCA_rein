import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
primary_color = '#4CAF50'  # Green for primary states
npca_color = '#2196F3'     # Blue for NPCA states
decision_color = '#FF9800' # Orange for decision points
tx_color = '#9C27B0'       # Purple for transmission states

# State boxes with rounded corners
states = {
    'PRIMARY_BACKOFF': (2, 8, primary_color),
    'PRIMARY_FROZEN': (2, 6, primary_color),
    'PRIMARY_TX': (2, 4, tx_color),
    'NPCA_BACKOFF': (8, 6, npca_color),
    'NPCA_FROZEN': (8, 4, npca_color),
    'NPCA_TX': (8, 2, tx_color),
    'DECISION': (5, 8, decision_color)
}

# Draw state boxes
state_boxes = {}
for state, (x, y, color) in states.items():
    if state == 'DECISION':
        # Diamond shape for decision point
        diamond = patches.RegularPolygon((x, y), 4, radius=0.8, 
                                       orientation=np.pi/4, 
                                       facecolor=color, 
                                       edgecolor='black', 
                                       linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, 'Decision\nPoint', ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    else:
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=2)
        ax.add_patch(box)
        # Format state names for display
        display_name = state.replace('_', '\n')
        ax.text(x, y, display_name, ha='center', va='center', 
                fontsize=9, fontweight='bold', color='white')
    state_boxes[state] = (x, y)

# Define transitions with labels
transitions = [
    # From PRIMARY_BACKOFF
    ('PRIMARY_BACKOFF', 'DECISION', 'OBSS detected', 'top'),
    ('PRIMARY_BACKOFF', 'PRIMARY_FROZEN', 'Intra-BSS busy', 'left'),
    ('PRIMARY_BACKOFF', 'PRIMARY_TX', 'Backoff=0 & idle', 'left'),
    
    # From DECISION POINT
    ('DECISION', 'PRIMARY_FROZEN', 'Action 0:\nStayPrimary', 'bottom'),
    ('DECISION', 'NPCA_BACKOFF', 'Action 1:\nGoNPCA', 'bottom'),
    
    # From PRIMARY_FROZEN
    ('PRIMARY_FROZEN', 'PRIMARY_BACKOFF', 'Channel idle', 'left'),
    
    # From PRIMARY_TX
    ('PRIMARY_TX', 'PRIMARY_BACKOFF', 'TX complete', 'left'),
    
    # From NPCA_BACKOFF
    ('NPCA_BACKOFF', 'NPCA_FROZEN', 'NPCA busy', 'right'),
    ('NPCA_BACKOFF', 'NPCA_TX', 'Backoff=0 & idle', 'right'),
    ('NPCA_BACKOFF', 'PRIMARY_BACKOFF', 'OBSS ended', 'bottom'),
    
    # From NPCA_FROZEN
    ('NPCA_FROZEN', 'NPCA_BACKOFF', 'NPCA idle', 'right'),
    ('NPCA_FROZEN', 'PRIMARY_BACKOFF', 'OBSS ended', 'bottom'),
    
    # From NPCA_TX
    ('NPCA_TX', 'NPCA_BACKOFF', 'TX complete &\nOBSS remains', 'right'),
    ('NPCA_TX', 'PRIMARY_BACKOFF', 'TX complete &\nOBSS ended', 'bottom'),
]

# Draw arrows
for start, end, label, side in transitions:
    start_x, start_y = state_boxes[start]
    end_x, end_y = state_boxes[end]
    
    # Adjust arrow start/end points based on state shapes
    if start == 'DECISION':
        if end_x > start_x:  # Going right
            start_x += 0.6
        elif end_x < start_x:  # Going left
            start_x -= 0.6
        if end_y < start_y:  # Going down
            start_y -= 0.6
    else:
        if end_x > start_x:  # Going right
            start_x += 0.8
            end_x -= 0.8
        elif end_x < start_x:  # Going left
            start_x -= 0.8
            end_x += 0.8
        if end_y > start_y:  # Going up
            start_y += 0.4
            end_y -= 0.4
        elif end_y < start_y:  # Going down
            start_y -= 0.4
            end_y += 0.4
    
    # Special handling for self-loops and curved arrows
    if start == end:
        # Self-loop
        continue
    elif (start == 'PRIMARY_BACKOFF' and end == 'PRIMARY_FROZEN') or \
         (start == 'PRIMARY_FROZEN' and end == 'PRIMARY_BACKOFF'):
        # Curved arrow for primary channel transitions
        connectionstyle = "arc3,rad=0.3"
    elif (start == 'NPCA_BACKOFF' and end == 'NPCA_FROZEN') or \
         (start == 'NPCA_FROZEN' and end == 'NPCA_BACKOFF'):
        # Curved arrow for NPCA channel transitions
        connectionstyle = "arc3,rad=-0.3"
    else:
        connectionstyle = "arc3,rad=0.1"
    
    # Draw arrow
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                          "data", "data",
                          arrowstyle="->", 
                          shrinkA=5, shrinkB=5,
                          connectionstyle=connectionstyle,
                          linewidth=2, 
                          color='black')
    ax.add_patch(arrow)
    
    # Add label
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    
    # Adjust label position based on arrow direction
    if side == 'top':
        mid_y += 0.3
    elif side == 'bottom':
        mid_y -= 0.3
    elif side == 'left':
        mid_x -= 0.5
    elif side == 'right':
        mid_x += 0.5
    
    ax.text(mid_x, mid_y, label, ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='white', 
                                edgecolor='gray', 
                                alpha=0.8))

# Add title and legends
ax.text(7, 9.5, 'Semi-MDP State Dynamics for NPCA Decision Making', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=primary_color, label='Primary Channel States'),
    patches.Patch(color=npca_color, label='NPCA Channel States'),
    patches.Patch(color=tx_color, label='Transmission States'),
    patches.Patch(color=decision_color, label='Decision Point')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Add state descriptions
descriptions = [
    "State Components (s_t):",
    "• OBSS remaining time (slots)",
    "• Radio transition time (slots)", 
    "• Transmission duration (slots)",
    "• Contention window index (0-6)",
    "",
    "Actions (a_t):",
    "• Action 0: StayPrimary",
    "• Action 1: GoNPCA",
    "",
    "Reward: Channel occupancy ratio",
    "at episode termination"
]

for i, desc in enumerate(descriptions):
    ax.text(11, 8.5 - i*0.3, desc, ha='left', va='center', fontsize=9,
            fontweight='bold' if desc.endswith(':') else 'normal')

plt.tight_layout()
plt.savefig('/Users/taewonsong/Code/NPCA_rein/semi_mdp_state_diagram.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Semi-MDP state diagram saved as 'semi_mdp_state_diagram.png'")