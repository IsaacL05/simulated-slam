from agent import agent
import numpy as np

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
                (5, 5),
                0.05)
print("10x10 array with landmarks set to 2:")
print(agent.map)
print(f"\nInterior landmarks set to 2: {agent.landmarks}")
print(f"Agent position set to 3: {agent.pos}")
print_estimates(agent)
print(agent.reward())
agent.update()
print_estimates(agent)
print(agent.reward())


for i in range(1000):
    random_value = np.random.random()

    if random_value < 0.25:
        agent.act('N')
    elif random_value < 0.5:
        agent.act('S')
    elif random_value < 0.75:
        agent.act('W')
    else:
        agent.act('E')
    
    agent.update()

    print_estimates(agent)
    print(agent.reward())

print_estimates(agent)
print(agent.pos_belief)
print(agent.landmarks_belief)