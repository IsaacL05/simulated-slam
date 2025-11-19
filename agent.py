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
        #self.map[self.pos[0], self.pos[1]] = 2
        # Previous position
        self.prev_pos = None

        # Most recent action
        self.prev_action = None

        # Agent's belief distribution over it's position (uniform prior)
        self.pos_belief = np.ones((10, 10))/100.0

        # Agent's belief distribution over the position of landmarks (uniform prior)
        self.landmarks_belief = np.ones((10, 10))/100.0

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
        
        # Store previous position before moving
        self.prev_pos = self.pos
        
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
        
        # Update position if target tile is not a landmark
        self.pos = (new_row, new_col)
        

    def _move_from_state(self, state, action):
        """
        Apply the deterministic motion model to a hypothetical agent state.

        Parameters
        ----------
        state : tuple[int, int]
            Candidate `(row, col)` index representing a possible agent location.
        action : str | None
            The control that was executed (`'N'`, `'S'`, `'E'`, `'W'`). When `None`
            (e.g., for the very first update), the state is returned unchanged.

        Returns
        -------
        tuple[int, int]
            The successor grid coordinate after applying the control. If the move
            would collide with a landmark, or the control is `None`, the original
            state is returned.
        """
        if action is None:
            return state
        row, col = state
        dr, dc = self._direction_vector(action)
        new_row = np.clip(row + dr, 0, 9)
        new_col = np.clip(col + dc, 0, 9)
        if self.map[new_row, new_col] == 1:
            return state
        return (new_row, new_col)

    def _predict_position_belief(self, prior_belief):
        """
        Propagate the prior position belief through the transition model.

        Implements the ∑_s T(s' | s, a) b(s) portion of the Bayes filter by
        pushing each prior mass element through `_move_from_state` and
        accumulating it at the resulting successor cell.
        """
        if self.prev_action is None:
            return prior_belief.copy()
        predicted = np.zeros_like(prior_belief)
        for row in range(10):
            for col in range(10):
                next_row, next_col = self._move_from_state((row, col), self.prev_action)
                predicted[next_row, next_col] += prior_belief[row, col]
        return predicted

    def _axis_measurement_prob(self, actual_idx, measured_idx):
        """
        Return P(measured_idx | actual_idx) for one grid axis under the ±1 lidar noise model.

        In the nominal case the reading is centered on the true index with probability
        (1 - 2 p_lidar_off) and shifts left/right by one cell with probability
        p_lidar_off each. If either offset would leave the grid, its mass is folded
        back into the on-grid reading so the total probability remains 1.
        """
        if measured_idx is None:
            return 1.0
        prob_correct = 1.0 - 2.0 * self.p_lidar_off
        prob_minus = self.p_lidar_off
        prob_plus = self.p_lidar_off

        minus_idx = actual_idx - 1
        plus_idx = actual_idx + 1

        if minus_idx < 0:
            # The -1 offset would leave the grid, so its probability mass is
            # reassigned to the on-target reading to keep the distribution normalized.
            prob_correct += prob_minus
            prob_minus = 0.0
        if plus_idx >= 10:
            # Likewise fold the +1 mass back when the ray would overshoot the grid.
            prob_correct += prob_plus
            prob_plus = 0.0

        probability_map = {actual_idx: prob_correct}
        if prob_minus > 0.0:
            # Only store side-offset probabilities when the corresponding cells exist.
            probability_map[minus_idx] = prob_minus
        if prob_plus > 0.0:
            probability_map[plus_idx] = prob_plus
        return probability_map.get(measured_idx, 0.0)

    def _compute_position_likelihood(self, row_est, col_est):
        """
        Build the observation likelihood for every grid cell based on noisy wall range readings.

        Each candidate cell `(row, col)` is weighted by the product of the independent
        row and column measurement probabilities returned by `_axis_measurement_prob`.
        Missing measurements (i.e., `None`) behave as uninformative factors equal to 1.
        """
        likelihood = np.ones((10, 10))
        if row_est is not None:
            for row in range(10):
                # Multiply each row slice by the 1D likelihood for the measured row
                # distance. Columns remain untouched because the axes factorize.
                likelihood[row, :] *= self._axis_measurement_prob(row, row_est)
        if col_est is not None:
            for col in range(10):
                # Mirror the same reasoning for the column dimension so that the
                # resulting grid encodes the product of independent axis beliefs.
                likelihood[:, col] *= self._axis_measurement_prob(col, col_est)
        return likelihood

    def _direction_vector(self, direction):
        """
        Map a cardinal action to its `(Δrow, Δcol)` displacement.

        Returns `(0, 0)` when the action is unrecognized so callers can safely
        fall back to a no-op move.
        """
        if direction == 'N':
            return -1, 0
        if direction == 'S':
            return 1, 0
        if direction == 'W':
            return 0, -1
        if direction == 'E':
            return 0, 1
        return 0, 0

    def _project(self, state, direction, distance):
        """
        Project a grid state forward by `distance` cells in `direction`.

        Unlike `_move_from_state`, this helper ignores collisions; callers must
        clamp or validate the result as needed.
        """
        row, col = state
        dr, dc = self._direction_vector(direction)
        return row + dr * distance, col + dc * distance

    def _is_within_grid(self, row, col):
        """Return True if the coordinates fall inside the 10x10 grid."""
        return 0 <= row < 10 and 0 <= col < 10

    def _landmark_aligned(self, state, landmark, direction):
        """
        Check whether the landmark lies along the ray cast from `state` in `direction`.

        The ray model assumes Manhattan-aligned scanning, so only cells that share
        the same row or column (and are "ahead" of the state) can be detected.
        """
        row, col = state
        l_row, l_col = landmark
        if direction == 'N':
            return l_col == col and l_row < row
        if direction == 'S':
            return l_col == col and l_row > row
        if direction == 'W':
            return l_row == row and l_col < col
        if direction == 'E':
            return l_row == row and l_col > col
        return False

    def _landmark_distance(self, state, landmark, direction):
        """
        Return the true distance to the landmark when it is aligned with the given ray.

        When the landmark is not visible in that direction, the helper returns `None`
        so that callers can short-circuit the likelihood computation.
        """
        if not self._landmark_aligned(state, landmark, direction):
            return None
        row, col = state
        l_row, l_col = landmark
        if direction in ('N', 'S'):
            return abs(row - l_row)
        if direction in ('W', 'E'):
            return abs(col - l_col)
        return None

    def _offset_out_of_bounds(self, state, direction, distance):
        """
        Indicate whether shifting the measurement by one more cell would leave the grid.

        Used to fold the one-sided `p_lidar_off` mass back onto the true range when the
        hypothetical over-estimate would cross the boundary.
        """
        target_row, target_col = self._project(state, direction, distance)
        return not self._is_within_grid(target_row, target_col)

    def _landmark_direction_likelihood(self, state, landmark, direction, observation):
        """
        Observation likelihood for a single direction given a candidate agent and landmark state.

        This captures the symmetric ±1 noise model for landmarks:
        - When a landmark is seen (`obs_type == 1`), the reported cell is the one located
          `distance` steps along `direction` from the agent. That cell receives weight
          `(1 - 2 p_lidar_off)` while its immediate neighbors along the same axis (±1 cell)
          receive `p_lidar_off` each. Off-grid neighbors fold their mass back to the
          reported cell.
        - When no landmark is detected (`obs_type == 0`), any candidate landmark that would
          have appeared within the reported empty distance is considered impossible.
        """
        distance, obs_type = observation
        if obs_type == 1:
            dr, dc = self._direction_vector(direction)
            expected = self._project(state, direction, distance)
            if not self._is_within_grid(*expected):
                return 0.0
            if not self._landmark_aligned(state, landmark, direction):
                return 0.0
            true_distance = self._landmark_distance(state, landmark, direction)
            if true_distance is None:
                return 0.0

            # Spread the detection's mass across the reported cell and its ±1
            # neighbors to represent range noise. Landmarks further down the ray
            # become "uninformative" (weight 1.0) so that detecting a nearer
            # landmark does not automatically eliminate them.
            prob_correct = 1.0 - 2.0 * self.p_lidar_off
            prob_minus = self.p_lidar_off
            prob_plus = self.p_lidar_off

            minus_cell = (expected[0] - dr, expected[1] - dc)
            plus_cell = (expected[0] + dr, expected[1] + dc)

            if not self._is_within_grid(*minus_cell):
                prob_correct += prob_minus
                prob_minus = 0.0
            if not self._is_within_grid(*plus_cell):
                prob_correct += prob_plus
                prob_plus = 0.0

            # If the landmark is closer than distance - 1, it should be impossible
            # (since we detected a landmark at distance, and landmarks at distance-1
            # are possible due to noise, but anything closer is impossible)
            if true_distance < distance - 1:
                return 0.0
            if true_distance > distance + 1:
                return 1.0
            if landmark == expected:
                return prob_correct
            if prob_minus > 0.0 and landmark == minus_cell:
                return prob_minus
            if prob_plus > 0.0 and landmark == plus_cell:
                return prob_plus
            return 0.0
        else:
            if not self._landmark_aligned(state, landmark, direction):
                # A missed detection offers no information about off-ray cells,
                # so their likelihood contribution remains neutral.
                return 1.0
            true_distance = self._landmark_distance(state, landmark, direction)
            if true_distance is None:
                return 1.0
            if true_distance <= distance:
                # The beam reported empty space up to `distance`; any landmark that
                # would have appeared before that distance is inconsistent.
                return 0.0
            return 1.0

    def _compute_landmark_likelihood(self, observations, state):
        """
        Evaluate landmark likelihoods assuming the agent occupies `state`.

        For each candidate landmark location `l'`, this reduces to

            O_L(o | s*, l') = Σ_{d ∈ {N,S,E,W}} P(o_d | s*, l', d)

        while discarding candidates that contradict an empty-beam observation.
        The sum reflects that independent directional detections may correspond
        to distinct physical landmarks.
        """
        likelihood = np.zeros((10, 10))
        for l_row in range(10):
            for l_col in range(10):
                cumulative_support = 0.0
                ruled_out = False
                for direction, observation in observations.items():
                    # Evaluate how this candidate aligns with each beam. The helper
                    # returns either a probability mass (for detections), 0 when the
                    # candidate contradicts the observation, or 1 for neutral evidence.
                    directional_prob = self._landmark_direction_likelihood(
                        state, (l_row, l_col), direction, observation
                    )
                    distance, obs_type = observation
                    if obs_type == 0:
                        if directional_prob == 0.0:
                            ruled_out = True
                            break
                        continue
                    # obs_type == 1: accumulate evidence from this direction
                    # If the cell is aligned with this direction and directional_prob is 0.0,
                    # it means the cell is too close (impossible), so rule it out
                    if self._landmark_aligned(state, (l_row, l_col), direction) and directional_prob == 0.0:
                        ruled_out = True
                        break
                    cumulative_support += directional_prob
                if ruled_out:
                    likelihood[l_row, l_col] = 0.0
                else:
                    # For all states for which their is no evidence either way, we give them a probability 1
                    # Landmarks that never receive positive support fall back to this
                    # neutral baseline, ensuring they stay in play for future updates.
                    likelihood[l_row, l_col] = cumulative_support if cumulative_support > 0.0 else 1.0
        return likelihood

    def update(self):
        observations = self.get_observations()
        row_est = None
        col_est = None

        # Lidar observations of walls
        if observations['N'][1] == 0:
            row_est = observations['N'][0]
        if observations['S'][1] == 0:
            row_est = 9 - observations['S'][0]
        if observations['W'][1] == 0:
            col_est = observations['W'][0]
        if observations['E'][1] == 0:
            col_est = 9 - observations['E'][0]

        # Bayes filter for pose: b'(s') propto O(o | a, s') * sum_s T(s' | s, a) b(s)
        # (eta, the normalizer, is applied by the division below).
        prior_pos_belief = self.pos_belief.copy()
        predicted_pos_belief = self._predict_position_belief(prior_pos_belief)
        pos_likelihood = self._compute_position_likelihood(row_est, col_est)
        self.pos_likelihood = pos_likelihood  # Store for visualization
        updated_pos_belief = predicted_pos_belief * pos_likelihood
        total_pos_prob = np.sum(updated_pos_belief)
        if total_pos_prob > 0:
            self.pos_belief = updated_pos_belief / total_pos_prob
        else:
            self.pos_belief = predicted_pos_belief

        # Landmark belief: approximate b_L'(l') propto O_L(o | s*, l') * b_L(l') with s* = argmax_s b'(s).
        most_likely_state = np.unravel_index(np.argmax(self.pos_belief), self.pos_belief.shape)
        landmark_likelihood = self._compute_landmark_likelihood(observations, most_likely_state)
        self.landmark_likelihood = landmark_likelihood  # Store for visualization
        updated_landmark_belief = self.landmarks_belief * landmark_likelihood
        total_landmark_prob = np.sum(updated_landmark_belief)
        if total_landmark_prob > 0:
            self.landmarks_belief = updated_landmark_belief / total_landmark_prob

    def reward(self):
        actual_agent = np.zeros_like(self.pos_belief, dtype=float)
        actual_agent[self.pos] = 1.0

        agent_diff = (self.pos_belief - actual_agent)
        agent_error = np.sum(agent_diff ** 2)

        actual_landmarks = (self.map == 1).astype(float) * 0.25
        landmark_diff = (self.landmarks_belief - actual_landmarks)
        landmark_error = np.sum(landmark_diff ** 2)

        return -(agent_error + landmark_error)
