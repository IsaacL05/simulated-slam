import numpy as np
import random

# Class representing an autonomous robot in a 2D grid world
class robot():
    def __init__(self, landmarks, start_pos, p_undershoot, p_overshoot):
        # Create a 12x12 grid world of all zeros (empty cells)
        self.map = np.zeros((12, 12), dtype=int)

        # Set all border cells to 1 (obstacles)
        self.map[0, :] = 1  # Top border
        self.map[-1, :] = 1  # Bottom border
        self.map[:, 0] = 1  # Left border
        self.map[:, -1] = 1  # Right border

        # Four positions set to 2 (landmarks)
        self.landmarks = landmarks

        for row, col in self.landmarks:
            self.map[row, col] = 2

        # Set robot position to 3 (robot starts in middle of map)
        self.pos = start_pos
        self.map[self.pos[0], self.pos[1]] = 3


        # Robot's belief distribution over it's position
        self.pos_belief = np.zeros((12, 12))

        # Set all border cells to 0 (robot can't be in a wall)
        self.pos_belief[0, :] = 0  # Top border
        self.pos_belief[-1, :] = 0  # Bottom border
        self.pos_belief[:, 0] = 0  # Left border
        self.pos_belief[:, -1] = 0  # Right border

        # Uniform prior over all non-wall positions
        for row in range (1, 11):
            for col in range(1, 11):
                self.pos_belief[row, col] = 0.01


        # Robot's belief distribution over the position of landmarks
        self.landmarks_belief = np.zeros((12, 12))

        # Set all border cells to 0 (obstacle can't be in a wall)
        self.landmarks_belief[0, :] = 0  # Top border
        self.landmarks_belief[-1, :] = 0  # Bottom border
        self.landmarks_belief[:, 0] = 0  # Left border
        self.landmarks_belief[:, -1] = 0  # Right border

        # Uniform prior over all non-wall positions
        for row in range (1, 11):
            for col in range(1, 11):
                self.landmarks_belief[row, col] = 0.01


        # Probabilities that a Lidar reading undershoots distance, overshoots distance, or measures correctly
        self.p_undershoot = p_undershoot
        self.p_overshoot = p_overshoot
        self.p_on_target = 1 - p_undershoot - p_overshoot


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
        
        # West (decreasing col)
        distance_w = 0
        obstacle_type_w = 0
        for c in range(col - 1, -1, -1):
            distance_w += 1
            if self.map[row, c] != 0:
                obstacle_type_w = self.map[row, c]
                break
        observations['W'] = (distance_w, obstacle_type_w)

        # East (increasing col)
        distance_e = 0
        obstacle_type_e = 0
        for c in range(col + 1, 12):
            distance_e += 1
            if self.map[row, c] != 0:
                obstacle_type_e = self.map[row, c]
                break
        observations['E'] = (distance_e, obstacle_type_e)
        
        return observations

    def update(self):
        observations = self.get_observations()
        row_est = None
        col_est = None
        landmarks = []

        # Lidar observations of walls
        if observations['N'][1] == 1:
            row_est = observations['N'][0]
        if observations['S'][1] == 1:
            row_est = 11 - observations['S'][0]
        if observations['W'][1] == 1:
            col_est = observations['W'][0]
        if observations['E'][1] == 1:
            col_est = 11 - observations['E'][0]

        # No noise for now - we're certain the lidars tell us the correct location
        self.pos_belief[row_est, col_est] = 1
        for row in range(12):
            for col in range(12):
                if row != row_est or col != col_est:
                    self.pos_belief[row, col] = 0

        # Lidar observations of landmarks
        if observations['N'][1] == 2:
            landmarks.append((row_est - observations['N'][0], col_est))
        if observations['S'][1] == 2:
            landmarks.append((row_est + observations['S'][0], col_est))
        if observations['W'][1] == 2:
            landmarks.append((row_est, col_est - observations['W'][0]))
        if observations['E'][1] == 2:
            landmarks.append((row_est, col_est - observations['E'][0]))

        # If landmarks are observed, we assume no noise for now and are certain of their placement
        if len(landmarks) != 0:
            for landmark in landmarks:
                self.landmarks_belief[landmark[0], landmark[1]] = 1/len(landmarks)
            for row in range(12):
                for col in range(12):
                    if (row, col) not in landmarks:
                        self.landmarks_belief[row, col] = 0

# Display the array
agent = robot([(3, 3), (4, 7), (7, 4), (8, 8)],
                (7, 7),
                0.05, 0.05)
print("12x12 array with borders set to 1, five random interior cells set to 1, and one cell set to 2:")
print(agent.map)
print(f"\nInterior landmarks set to 2: {agent.landmarks}")
print(f"Robot position set to 3: {agent.pos}")
print(agent.pos_belief)
print(agent.landmarks_belief)
agent.update()
print(agent.pos_belief)
print(agent.landmarks_belief)
