from agent import agent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def print_estimates(agent):
    # Print index of greatest element in pos_belief
    max_pos_idx = np.unravel_index(np.argmax(agent.pos_belief), agent.pos_belief.shape)
    print(f"\nIndex of greatest element in pos_belief: {max_pos_idx}")

    # Print 4 greatest elements (indices) in landmarks_belief
    # Get indices of top 4 values, sorted from highest to lowest
    flat_landmarks = agent.landmarks_belief.flatten()
    top_4_flat_indices = np.argpartition(flat_landmarks, -4)[-4:]
    top_4_flat_indices_sorted = top_4_flat_indices[np.argsort(flat_landmarks[top_4_flat_indices])[::-1]]
    top_4_indices_2d = [np.unravel_index(idx, agent.landmarks_belief.shape) for idx in top_4_flat_indices_sorted]
    print(f"4 greatest elements in landmarks_belief (indices): {top_4_indices_2d}")

agent = agent([(2, 2), (3, 6), (6, 3), (7, 7)],
                (6, 6),
                0.1)

# Create visualization grid
def visualize_grid(agent):
    """Visualize the 10x10 grid with agent and landmark positions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a grid visualization array (empty cells = 0, landmarks = 1, agent = 2)
    grid_vis = np.zeros((10, 10))
    
    # Mark landmarks (stored as (row, col) in agent.landmarks)
    for row, col in agent.landmarks:
        grid_vis[row, col] = 1
    
    # Mark agent position (stored as (row, col) in agent.pos)
    agent_row, agent_col = agent.pos
    grid_vis[agent_row, agent_col] = 2
    
    # Create colormap: empty=white, landmarks=red, agent=blue
    colors = ['white', 'red', 'blue']
    bounds = [0, 1, 2, 3]
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Display grid
    im = ax.imshow(grid_vis, cmap=cmap, norm=norm, aspect='equal', 
                   extent=[-0.5, 9.5, 9.5, -0.5], interpolation='nearest')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    
    # Label x coordinates (rows) on left
    ax.set_yticks(np.arange(0, 10))
    ax.set_yticklabels([str(i) for i in range(10)], fontsize=14)
    ax.set_ylabel('x', fontsize=16)
    
    # Label y coordinates (columns) on top
    ax.set_xticks(np.arange(0, 10))
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=14)
    ax.set_xlabel('y', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    # Add legend in upper right
    landmark_patch = mpatches.Patch(color='red', label='Landmark')
    agent_patch = mpatches.Patch(color='blue', label='Agent')
    ax.legend(handles=[landmark_patch, agent_patch], 
              loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=14)
    
    plt.tight_layout()
    plt.show()

def visualize_likelihood(likelihood):
    """Visualize a likelihood grid with darker colors for higher probabilities and numeric values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use grayscale colormap (darker = higher probability)
    im = ax.imshow(likelihood, cmap='gray_r', aspect='equal', 
                   extent=[-0.5, 9.5, 9.5, -0.5], interpolation='nearest')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Label x coordinates (rows) on left
    ax.set_yticks(np.arange(0, 10))
    ax.set_yticklabels([str(i) for i in range(10)], fontsize=14)
    ax.set_ylabel('x', fontsize=16)
    
    # Label y coordinates (columns) on top
    ax.set_xticks(np.arange(0, 10))
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=14)
    ax.set_xlabel('y', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    # Add numeric values in each cell
    for row in range(10):
        for col in range(10):
            value = likelihood[row, col]
            # Use white text for darker cells, black for lighter cells
            text_color = 'white' if value > likelihood.max() / 2 else 'black'
            ax.text(col, row, f'{value:.3f}', ha='center', va='center', 
                   fontsize=10, color=text_color, weight='bold')
    
    plt.tight_layout()
    plt.show()

# Visualize the grid
visualize_grid(agent)

print("10x10 array with landmarks set to 2:")
print(agent.map)
print(f"\nInterior landmarks set to 2: {agent.landmarks}")
print(f"Agent position set to 3: {agent.pos}")
print_estimates(agent)
print(agent.reward())
agent.update()

# Visualize likelihoods after first update
visualize_likelihood(agent.pos_likelihood)
visualize_likelihood(agent.landmark_likelihood)

# print_estimates(agent)
# print(agent.reward())


# for i in range(1000):
#     random_value = np.random.random()

#     if random_value < 0.25:
#         agent.act('N')
#     elif random_value < 0.5:
#         agent.act('S')
#     elif random_value < 0.75:
#         agent.act('W')
#     else:
#         agent.act('E')
    
#     agent.update()

#     print_estimates(agent)
#     print(agent.reward())

# print_estimates(agent)
# print(agent.pos_belief)
# print(agent.landmarks_belief)