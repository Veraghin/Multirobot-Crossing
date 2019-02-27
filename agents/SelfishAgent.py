from agents.Agent import Agent
import numpy as np


class SelfishAgent(Agent):
    # Controller for the process of choosing the target velocity for the next
    # time step
    def choose_target_velocity(self, neighbours):
        if self.at_goal():
            self.target_velocity = np.zeros(2, dtype=np.float32)
            return

        offset_to_goal = self._goal - self.get_position()
        dist_to_goal = np.linalg.norm(offset_to_goal)

        if self.preferred_speed * self.dt > dist_to_goal:
            # Can reach goal in one step
            target_velocity = offset_to_goal / self.dt
        else:
            # Want to move at preferred speed in unit direction of goal
            target_velocity = self.preferred_speed * \
                              offset_to_goal / dist_to_goal

        if self.holonomic:
            self.target_velocity = target_velocity
        else:
            self.target_velocity = self.compute_wheel_speeds(target_velocity)
