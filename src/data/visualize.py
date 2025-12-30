import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json

# Official ARC Color Scheme
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Pink/Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Light Blue
    "#870C25",  # 9: Maroon
]

def plot_grid(ax, grid, title=""):
    """
    Helper to plot a single ARC grid on a matplotlib axis.
    """
    grid = np.array(grid)
    h, w = grid.shape
    
    # Create a custom colormap for discrete 0-9 values
    cmap = colors.ListedColormap(ARC_COLORS)
    bounds = list(range(11)) # 0 to 10
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the grid
    ax.imshow(grid, cmap=cmap, norm=norm)
    
    # Grid lines (draw distinct lines between pixels)
    ax.grid(which='major', axis='both', linestyle='-', color='#555555', linewidth=1)
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis tick marks (ticks) but keep the grid lines
    ax.tick_params(axis='both', which='both', length=0)
    
    ax.set_title(f"{title}\n({h}x{w})", fontsize=10)

def visualize_puzzle(puzzle_data, aug_id=None):
    """
    Visualizes a full ARC puzzle (Train pairs + Test pairs).
    
    Args:
        puzzle_data (dict): Dictionary containing 'train' and 'test' lists.
        aug_id (str): Optional title/ID to display at the top.
    """
    train_pairs = puzzle_data.get('train', [])
    test_pairs = puzzle_data.get('test', [])
    
    total_pairs = len(train_pairs) + len(test_pairs)
    
    # Create Figure: Rows = Pairs, Cols = 2 (Input, Output)
    fig, axes = plt.subplots(total_pairs, 2, figsize=(8, 3 * total_pairs))
    
    # Handle case of single pair (subplots returns 1D array)
    if total_pairs == 1:
        axes = np.array([axes])
    
    if aug_id:
        fig.suptitle(f"Puzzle: {aug_id}", fontsize=14, fontweight='bold')
    
    current_row = 0
    
    # --- PLOT TRAIN PAIRS ---
    for i, pair in enumerate(train_pairs):
        ax_in = axes[current_row, 0]
        ax_out = axes[current_row, 1]
        
        plot_grid(ax_in, pair['input'], title=f"Train {i+1} Input")
        plot_grid(ax_out, pair['output'], title=f"Train {i+1} Output")
        
        current_row += 1
        
    # --- PLOT TEST PAIRS ---
    for i, pair in enumerate(test_pairs):
        ax_in = axes[current_row, 0]
        ax_out = axes[current_row, 1]
        
        plot_grid(ax_in, pair['input'], title=f"TEST {i+1} Input")
        
        if 'output' in pair:
            plot_grid(ax_out, pair['output'], title=f"TEST {i+1} Target")
        else:
            # If test output is hidden (actual inference)
            ax_out.text(0.5, 0.5, "?", fontsize=40, ha='center', va='center')
            ax_out.set_title(f"TEST {i+1} Output")
            ax_out.axis('off')
            
        # Add a colored border or background to distinguish Test
        # (Optional aesthetic choice)
        
        current_row += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    plt.show()


def visualize_puzzle_from_file(filepath):
    with open(filepath, 'r') as fd:
        puzzle_data = json.loads(fd.read())
    visualize_puzzle(puzzle_data)
