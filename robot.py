import numpy as np
import random

# Class representing an autonomous robot in a 2D grid world
class robot():
    def __init__(self):
        # Create a 12x12 grid world of all zeros (empty cells)
        self.map = np.zeros((12, 12), dtype=int)

        # Set all border cells to 1 (obstacles)
        self.map[0, :] = 1  # Top border
        self.map[-1, :] = 1  # Bottom border
        self.map[:, 0] = 1  # Left border
        self.map[:, -1] = 1  # Right border

        # Choose five random unique cells from interior (non-border) and set them to 2 (obstacles)
        interior_positions = [(i, j) for i in range(1, 11) for j in range(1, 11)]
        self.obstacles = random.sample(interior_positions, 5)

        for row, col in self.obstacles:
            self.map[row, col] = 2

        # Pick one more random cell from interior and set it to 3 (robot position)
        remaining_positions = [pos for pos in interior_positions if pos not in self.obstacles]
        self.pos = random.choice(remaining_positions)
        self.map[self.pos[0], self.pos[1]] = 3


        # Robot's belief distribution over it's position
        self.pos_belief = np.zeros((12, 12))

        # Set all border cells to 0 (robot can't be in a wall)
        self.pos_belief[0, :] = 0  # Top border
        self.pos_belief[-1, :] = 0  # Bottom border
        self.pos_belief[:, 0] = 0  # Left border
        self.pos_belief[:, -1] = 0  # Right border

        # Uniform prior over all non-wall positions
        for row, col in interior_positions:
            self.pos_belief[row, col] = 0.01


        # Robot's belief distribution over the position of each landmark
        self.obstacles_belief = []

        for i in range(5):
            self.obstacles_belief.append(np.zeros((12, 12)))

            # Set all border cells to 0 (obstacle can't be in a wall)
            self.obstacles_belief[i][0, :] = 0  # Top border
            self.obstacles_belief[i][-1, :] = 0  # Bottom border
            self.obstacles_belief[i][:, 0] = 0  # Left border
            self.obstacles_belief[i][:, -1] = 0  # Right border

            # Uniform prior over all non-wall positions
            for row, col in interior_positions:
                self.obstacles_belief[i][row, col] = 0.01


    def get_observations(self):
        # Simulated Lidar in 4 cardinal directions
        # First parameter: distance to object
        # Second parameter: type of obstruction (1 = wall, 2 = obstacle)
        row, col = self.pos
        observations = {}
        
        # North (decreasing row)
        distance_n = 0
        obstacle_type_n = 0
        for r in range(row - 1, -1, -1):
            distance_n += 1
            if self.map[r, col] != 0:
                obstacle_type_n = self.map[r, col]
                break
        observations['N'] = (distance_n, obstacle_type_n)
        
        # South (increasing row)
        distance_s = 0
        obstacle_type_s = 0
        for r in range(row + 1, 12):
            distance_s += 1
            if self.map[r, col] != 0:
                obstacle_type_s = self.map[r, col]
                break
        observations['S'] = (distance_s, obstacle_type_s)
        
        # East (increasing col)
        distance_e = 0
        obstacle_type_e = 0
        for c in range(col + 1, 12):
            distance_e += 1
            if self.map[row, c] != 0:
                obstacle_type_e = self.map[row, c]
                break
        observations['E'] = (distance_e, obstacle_type_e)
        
        # West (decreasing col)
        distance_w = 0
        obstacle_type_w = 0
        for c in range(col - 1, -1, -1):
            distance_w += 1
            if self.map[row, c] != 0:
                obstacle_type_w = self.map[row, c]
                break
        observations['W'] = (distance_w, obstacle_type_w)
        
        return observations

# Display the array
agent = robot()
print("12x12 array with borders set to 1, five random interior cells set to 1, and one cell set to 2:")
print(agent.map)
print(f"\nInterior obstacles set to 2: {agent.obstacles}")
print(f"Robot position set to 3: {agent.pos}")
#print(agent.pos_belief)
#print(agent.obstacles_belief[0])
print(agent.get_observations())
