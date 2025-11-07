import numpy as np

# Class representing an autonomous agent in a 2D grid world
class agent():
    def __init__(self, landmarks, start_pos, p_lidar_off):
        # Create a 10x10 grid world of all zeros (empty cells)
        self.map = np.zeros((10, 10), dtype=int)

        # Four positions set to 1 (landmarks)
        self.landmarks = landmarks

        for row, col in self.landmarks:
            self.map[row, col] = 1

        # Set agent position to 2
        self.pos = start_pos
        self.map[self.pos[0], self.pos[1]] = 2


        # Agent's belief distribution over it's position
        self.pos_belief = np.zeros((10, 10))

        # Uniform prior over all positions
        for row in range (10):
            for col in range(10):
                self.pos_belief[row, col] = 0.01


        # Agent's belief distribution over the position of landmarks
        self.landmarks_belief = np.zeros((10, 10))

        # Uniform prior over all positions
        for row in range (10):
            for col in range(10):
                self.landmarks_belief[row, col] = 0.01


        # Probability that observation from Lidar measurements is 1 tile off (each direction)
        self.p_lidar_off = p_lidar_off
        self.p_on_target = 1 - 4 * p_lidar_off


    def get_observations(self):
        # Simulated Lidar in 4 cardinal directions
        # First parameter: distance to object
        # Second parameter: type of obstruction (1 = landmark, 0 = empty)
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
        for r in range(row + 1, 10):
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
        for c in range(col + 1, 10):
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
        if observations['N'][1] == 0:
            row_est = observations['N'][0]
        if observations['S'][1] == 0:
            row_est = 9 - observations['S'][0]
        if observations['W'][1] == 0:
            col_est = observations['W'][0]
        if observations['E'][1] == 0:
            col_est = 9 - observations['E'][0]

        # No noise for now - we're certain the lidars tell us the correct location
        self.pos_belief[row_est, col_est] = 1
        for row in range(10):
            for col in range(10):
                if row != row_est or col != col_est:
                    self.pos_belief[row, col] = 0

        # Lidar observations of landmarks
        if observations['N'][1] == 1:
            landmarks.append((row_est - observations['N'][0], col_est))
        if observations['S'][1] == 1:
            landmarks.append((row_est + observations['S'][0], col_est))
        if observations['W'][1] == 1:
            landmarks.append((row_est, col_est - observations['W'][0]))
        if observations['E'][1] == 1:
            landmarks.append((row_est, col_est - observations['E'][0]))

        # If landmarks are observed, we assume no noise for now and are certain of their placement
        if len(landmarks) != 0:
            for landmark in landmarks:
                self.landmarks_belief[landmark[0], landmark[1]] = 1/len(landmarks)
            for row in range(10):
                for col in range(10):
                    if (row, col) not in landmarks:
                        self.landmarks_belief[row, col] = 0

# Display the array
agent  = agent([(2, 2), (3, 6), (6, 3), (7, 7)],
                (5, 5),
                0.025)
print("10x10 array with landmarks set to 2:")
print(agent.map)
print(f"\nInterior landmarks set to 2: {agent.landmarks}")
print(f"Agent position set to 3: {agent.pos}")
print(agent.pos_belief)
print(agent.landmarks_belief)
agent.update()
print(agent.pos_belief)
print(agent.landmarks_belief)
