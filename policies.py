import numpy as np

def planning_policy(agent, collision_penalty=1.0):
    """
    Policy that uses online lookahead with rollouts to select the best action.
    """
    return agent.plan(collision_penalty=collision_penalty)


def random_policy(agent):
    """
    Baseline policy that selects a random action uniformly.
    """
    actions = ['N', 'S', 'E', 'W']
    return np.random.choice(actions)
