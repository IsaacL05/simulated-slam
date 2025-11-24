import numpy as np
from agent import agent

def run_episode(agent, policy_fn, num_steps, collision_penalty=1.0, verbose=False):
    """
    Run a single episode using the given policy.

    Returns:
    - Dictionary containing episode metrics:
        - total_reward: Cumulative reward over the episode
        - rewards: List of rewards at each step
        - actions: List of actions taken
        - collisions: Number of collisions
        - final_pos_error: Final position belief error
        - final_landmark_error: Final landmark belief error
    """
    rewards = []
    actions = []
    collisions = 0

    for step in range(num_steps):
        # Select action using the policy
        action = policy_fn(agent)
        actions.append(action)

        # Execute action
        agent.act(action)

        # Track collisions
        if agent.collision_occurred:
            collisions += 1

        # Update beliefs
        agent.update()

        # Calculate reward
        reward = agent.reward(collision_penalty=collision_penalty)
        rewards.append(reward)

        if verbose and (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{num_steps}: reward = {reward:.4f}")

    # Calculate final belief errors
    # Only compute error at agent position and four landmark positions
    # Beliefs are scaled by 100, so perfect belief would be 100 at agent position
    agent_pos_belief = agent.pos_belief[agent.pos[0], agent.pos[1]]
    agent_actual = 100.0
    final_pos_error = (agent_pos_belief - agent_actual) ** 2

    # Beliefs are scaled by 100, so perfect belief would be 25 at each landmark (25/100 = 0.25)
    final_landmark_error = 0.0
    landmark_actual = 25.0  # Each landmark has probability 0.25 (1/4), scaled by 100 = 25
    for landmark_row, landmark_col in agent.landmarks:
        landmark_pos_belief = agent.landmarks_belief[landmark_row, landmark_col]
        final_landmark_error += (landmark_pos_belief - landmark_actual) ** 2

    return {
        'total_reward': sum(rewards),
        'mean_reward': np.mean(rewards),
        'rewards': rewards,
        'actions': actions,
        'collisions': collisions,
        'final_pos_error': final_pos_error,
        'final_landmark_error': final_landmark_error,
        'final_total_error': final_pos_error + final_landmark_error
    }


def compare_policies(landmarks, start_pos, p_lidar_off, num_steps,
                     num_trials=5, planning_params=None, collision_penalty=1.0,
                     verbose=False):
    """
    Compare planning policy vs random policy over multiple trials.

    Returns:
    - Dictionary with comparison results
    """
    from policies import planning_policy, random_policy

    if planning_params is None:
        planning_params = {'num_rollouts': 10, 'horizon': 10, 'gamma': 0.95}

    planning_results = []
    random_results = []

    for trial in range(num_trials):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial + 1}/{num_trials}")
            print(f"{'='*60}")

        # Run planning policy
        if verbose:
            print(f"\n--- Planning Policy ---")
        planning_agent = agent(landmarks, start_pos, p_lidar_off,
                              num_rollouts=planning_params['num_rollouts'],
                              horizon=planning_params['horizon'],
                              gamma=planning_params['gamma'])
        planning_result = run_episode(planning_agent, planning_policy, num_steps,
                                     collision_penalty, verbose=verbose)
        planning_results.append(planning_result)

        # Run random policy
        if verbose:
            print(f"\n--- Random Policy ---")
        random_agent = agent(landmarks, start_pos, p_lidar_off,
                           num_rollouts=1, horizon=1, gamma=0.95)  # Dummy params for random
        random_result = run_episode(random_agent, random_policy, num_steps,
                                   collision_penalty, verbose=verbose)
        random_results.append(random_result)

    # Aggregate results
    def aggregate_metrics(results):
        return {
            'mean_total_reward': np.mean([r['total_reward'] for r in results]),
            'std_total_reward': np.std([r['total_reward'] for r in results]),
            'mean_collisions': np.mean([r['collisions'] for r in results]),
            'std_collisions': np.std([r['collisions'] for r in results]),
            'mean_final_pos_error': np.mean([r['final_pos_error'] for r in results]),
            'std_final_pos_error': np.std([r['final_pos_error'] for r in results]),
            'mean_final_landmark_error': np.mean([r['final_landmark_error'] for r in results]),
            'std_final_landmark_error': np.std([r['final_landmark_error'] for r in results]),
            'mean_final_total_error': np.mean([r['final_total_error'] for r in results]),
            'std_final_total_error': np.std([r['final_total_error'] for r in results]),
        }

    planning_stats = aggregate_metrics(planning_results)
    random_stats = aggregate_metrics(random_results)

    return {
        'planning': planning_stats,
        'random': random_stats,
        'planning_results': planning_results,
        'random_results': random_results
    }


def print_comparison(comparison_results):
    """
    Print a formatted comparison of policy results.
    """
    planning = comparison_results['planning']
    random = comparison_results['random']

    print("\n" + "="*70)
    print("POLICY COMPARISON RESULTS")
    print("="*70)

    print(f"\n{'Metric':<30} {'Planning':<20} {'Random':<20}")
    print("-"*70)

    print(f"{'Total Reward':<30} {planning['mean_total_reward']:>8.4f} ± {planning['std_total_reward']:<8.4f} "
          f"{random['mean_total_reward']:>8.4f} ± {random['std_total_reward']:<8.4f}")

    print(f"{'Collisions':<30} {planning['mean_collisions']:>8.2f} ± {planning['std_collisions']:<8.2f} "
          f"{random['mean_collisions']:>8.2f} ± {random['std_collisions']:<8.2f}")

    print(f"{'Final Position Error':<30} {planning['mean_final_pos_error']:>8.4f} ± {planning['std_final_pos_error']:<8.4f} "
          f"{random['mean_final_pos_error']:>8.4f} ± {random['std_final_pos_error']:<8.4f}")

    print(f"{'Final Landmark Error':<30} {planning['mean_final_landmark_error']:>8.4f} ± {planning['std_final_landmark_error']:<8.4f} "
          f"{random['mean_final_landmark_error']:>8.4f} ± {random['std_final_landmark_error']:<8.4f}")

    print(f"{'Final Total Error':<30} {planning['mean_final_total_error']:>8.4f} ± {planning['std_final_total_error']:<8.4f} "
          f"{random['mean_final_total_error']:>8.4f} ± {random['std_final_total_error']:<8.4f}")

    print("\n" + "="*70)

    # Calculate improvements
    reward_improvement = ((planning['mean_total_reward'] - random['mean_total_reward']) /
                         abs(random['mean_total_reward']) * 100)
    collision_reduction = ((random['mean_collisions'] - planning['mean_collisions']) /
                          max(random['mean_collisions'], 1) * 100)
    error_reduction = ((random['mean_final_total_error'] - planning['mean_final_total_error']) /
                      random['mean_final_total_error'] * 100)

    print(f"Planning vs Random:")
    print(f"  Reward improvement: {reward_improvement:+.2f}%")
    print(f"  Collision reduction: {collision_reduction:+.2f}%")
    print(f"  Total error reduction: {error_reduction:+.2f}%")
    print("="*70 + "\n")
