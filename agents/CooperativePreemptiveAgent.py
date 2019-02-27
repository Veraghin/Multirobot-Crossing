from agents.CooperativeAgent import CooperativeAgent, Candidate
from agents.agent_utils import det, normal
import numpy as np

class CooperativePreemptiveAgent(CooperativeAgent):
    
    historical_headings = dict()
    malicious = None

    # Controller tries to find the malicious robot based on it's heading
    # Assumes no gap in communication between robots
    def try_to_find_malicious_agent(self, neighbours):
        enough_data = True
        if len(neighbours) == 0:
            enough_data = False
        for n in neighbours:
            name = n.get_id()
            heading = n.get_velocity()
            if name in self.historical_headings.keys():
                self.historical_headings[name] = self.historical_headings[name] + heading
                if len(self.historical_headings[name]) < 20:
                    enough_data = False
            else:
                self.historical_headings[name] = [heading]
                enough_data = False
        # If there is a sufficient heading information available
        if enough_data:
            smallest_change = 50.0
            mal = None
            for n, headings in self.historical_headings.items():
                change = 0.0
                for i in range(len(headings)-1):
                    change += np.linalg.norm(headings[i+1] - headings[i])
                if change < smallest_change:
                    smallest_change = change
                    mal = n
            self.malicious = neighbours[mal]
        else:
            self.malicious = None

    # Controller for the process of choosing the target velocity for the next
    # time step
    def choose_target_velocity(self, neighbours):
        if self.at_goal():
            self.target_velocity = np.zeros(2, dtype=np.float32)
            return

        position = self.get_position()
        velocity = self.get_velocity()
        preferred_velocity = self.choose_preferred_velocity()

        if self.malicious is None or len(neighbours) != len(self.historical_headings.keys()):
            self.try_to_find_malicious_agent(neighbours)
            if self.malicious is not None:
                del neighbours[self.malicious.get_id()]

        # Compute velocity obstacles for each neighbouring robot
        velocity_obstacles = self.compute_hrvos(
            position, velocity, neighbours[:])
        if self.malicious is not None:
            # compute vo for it
            pass

        # Create the set of all possible candidate points that we can choose to
        # move towards using the ClearPath algorithm

        # If no obstacles, candidate is to move straight in desired direction
        if np.linalg.norm(preferred_velocity) < self.max_speed:
            pos = preferred_velocity
        else:
            pos = self.max_speed * preferred_velocity / np.linalg.norm(
                preferred_velocity)

        candidates = [(
            np.linalg.norm(preferred_velocity - pos),
            Candidate(pos, -1, -1))]

        # Compute candidates using HRVOs
        candidates.extend(
            self.compute_projection_candidates(velocity_obstacles,
                                               preferred_velocity))
        candidates.extend(
            self.compute_max_candidates(velocity_obstacles, preferred_velocity))
        candidates.extend(
            self.compute_intersect_candidates(velocity_obstacles,
                                              preferred_velocity))

        # Choose the best candidate point that is closest to the desired
        # velocity towards the goal but will not cause any collisions or if no
        # such point exists, the point that will not collide with the largest
        # n closest robots

        # Start by finding all candidates that avoid the maximum n nearest
        # robots
        optimal_candidates = self.find_optimal_candidates(candidates,
                                                          velocity_obstacles)

        # Of all the optimal candidates, find the one that is closest to the
        # desired velocity
        self.target_velocity = self.choose_best_velocity(optimal_candidates)
