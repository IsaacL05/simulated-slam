import numpy as np

# Class representing an autonomous agent in a 2D grid world
class agent():
    def __init__(self, landmarks, start_pos, p_stay, p_lidar_off):
        # Create a 10x10 grid world of all zeros (empty cells)
        self.map = np.zeros((10, 10), dtype=int)

        # Four positions set to 1 (landmarks)
        self.landmarks = landmarks

        for row, col in self.landmarks:
            self.map[row, col] = 1

        # Set agent position to 2
        self.pos = start_pos
        #self.map[self.pos[0], self.pos[1]] = 2
        # Previous position
        self.prev_pos = None

        # Most recent action
        self.prev_action = None

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


        #Probability that the agent stays in place when trying to move
        self.p_stay = p_stay

        # Probability that observation from Lidar measurements is 1 tile off (each direction)
        self.p_lidar_off = p_lidar_off


    def get_observations(self):
        # Simulated lidar in 4 cardinal directions
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

    def act(self, direction):
        self.prev_action = direction

        # Generate a random number between 0 and 1
        random_value = np.random.random()
        
        # Probabilistic if statement: stay in place with probability p_stay
        if random_value < self.p_stay:
            # Agent stays in place
            return
        
        # Agent moves in the specified direction
        row, col = self.pos
        
        # Update position based on direction
        if direction == 'N':
            new_row = max(0, row - 1)  # Move north (decrease row)
            new_col = col
        elif direction == 'S':
            new_row = min(9, row + 1)  # Move south (increase row)
            new_col = col
        elif direction == 'W':
            new_row = row
            new_col = max(0, col - 1)  # Move west (decrease col)
        else:
            new_row = row
            new_col = min(9, col + 1)  # Move east (increase col)
        
        # Check if the target tile is a landmark - if so, don't move
        if self.map[new_row, new_col] == 1:
            # Target tile is a landmark, agent stays in place
            return
        
        # Store previous position before moving (only if we're actually moving)
        self.prev_pos = (row, col)
        
        # Update position if target tile is not a landmark
        self.pos = (new_row, new_col)
        

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
