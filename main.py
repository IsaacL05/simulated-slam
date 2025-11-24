from numpy.random import random_sample
from agent import agent
from policies import planning_policy, random_policy
from evaluation import run_episode, compare_policies, print_comparison
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
    # Fix scale so 0 = white and 1 = black regardless of data range
    im = ax.imshow(likelihood, cmap='gray_r', aspect='equal',
                   extent=[-0.5, 9.5, 9.5, -0.5], interpolation='nearest',
                   vmin=0, vmax=1)

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
            # Use white text for darker cells (value > 0.5), black for lighter cells
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                   fontsize=10, color=text_color, weight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Environment setup
    landmarks = [(2, 2), (3, 6), (6, 3), (7, 7)]
    # Generate all valid positions (0-9, 0-9) excluding landmarks
    valid_positions = [(r, c) for r in range(10) for c in range(10) if (r, c) not in landmarks]
    start_pos = tuple(valid_positions[np.random.randint(len(valid_positions))])
    p_lidar_off = 0.1

    # Planning hyperparameters
    planning_params = {
        'num_rollouts': 5,
        'horizon': 5,
        'gamma': 0.95
    }

    collision_penalty = 1.0
    num_steps = 100
    num_trials = 3

    print("="*70)
    print("SLAM POLICY COMPARISON")
    print("="*70)
    print(f"\nEnvironment:")
    print(f"  Landmarks: {landmarks}")
    print(f"  Start position: {start_pos}")
    print(f"  Lidar noise: {p_lidar_off}")
    print(f"\nPlanning parameters:")
    print(f"  Rollouts: {planning_params['num_rollouts']}")
    print(f"  Horizon: {planning_params['horizon']}")
    print(f"  Gamma: {planning_params['gamma']}")
    print(f"\nEvaluation:")
    print(f"  Steps per episode: {num_steps}")
    print(f"  Number of trials: {num_trials}")
    print(f"  Collision penalty: {collision_penalty}")

    # Run comparison
    print("\nRunning comparison...")
    results = compare_policies(
        landmarks=landmarks,
        start_pos=start_pos,
        p_lidar_off=p_lidar_off,
        num_steps=num_steps,
        num_trials=num_trials,
        planning_params=planning_params,
        collision_penalty=collision_penalty,
        verbose=False
    )

    # Print results
    print_comparison(results)

    # Optional: Visualize a single episode with planning
    print("\n" + "="*70)
    print("Running single episode with PLANNING POLICY for visualization...")
    print("="*70)

    vis_agent = agent(landmarks, start_pos, p_lidar_off,
                     num_rollouts=planning_params['num_rollouts'],
                     horizon=planning_params['horizon'],
                     gamma=planning_params['gamma'])

    print("\nInitial state:")
    visualize_grid(vis_agent)
    print_estimates(vis_agent)
    print(f"Initial reward: {vis_agent.reward(collision_penalty)}")

    print("\nRunning 100 steps with planning policy...")
    episode_result = run_episode(vis_agent, planning_policy, 100,
                                 collision_penalty=collision_penalty, verbose=True)

    print(f"\nFinal state after 100 steps:")
    print_estimates(vis_agent)
    print(f"Episode total reward: {episode_result['total_reward']:.4f}")
    print(f"Episode collisions: {episode_result['collisions']}")

    visualize_grid(vis_agent)
    visualize_likelihood(vis_agent.pos_belief/100)
    visualize_likelihood(vis_agent.landmarks_belief/100)