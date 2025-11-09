from agent import agent

agent = agent([(2, 2), (3, 6), (6, 3), (7, 7)],
                (5, 5),
                0.05,
                0.1)
print("10x10 array with landmarks set to 2:")
print(agent.map)
print(f"\nInterior landmarks set to 2: {agent.landmarks}")
print(f"Agent position set to 3: {agent.pos}")
print(agent.pos_belief)
print(agent.landmarks_belief)
agent.update()
print(agent.pos_belief)
print(agent.landmarks_belief)
agent.act('N')
agent.act('N')
agent.update()
print(agent.pos_belief)
print(agent.landmarks_belief)
