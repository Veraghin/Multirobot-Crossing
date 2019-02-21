from abc import ABC, abstractmethod
import numpy as np

EPSILON = 1e-3


class Agent(ABC):
    def __init__(self, name, position, goal, radius=0.3, simulation_step=0.1,
                 preferred_speed=0.5, max_speed=1., noise=0):
        self.id = name
        self._ground_pose = position
        self.measured_pose = position
        self._goal = goal
        self.target_velocity = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.measured_velocity = np.zeros(2, dtype=np.float32)
        self.dt = simulation_step
        self.noise = noise
        self.radius = radius
        self.preferred_speed = preferred_speed
        self.max_speed = 1.
        self.preferred_velocity = np.zeros(2, dtype=np.float32)

    def get_position(self):
        return self.measured_pose

    def set_position(self, pose):
        self._ground_pose = pose

    def get_velocity(self):
        return self.measured_velocity

    def at_goal(self):
        return np.linalg.norm(self._goal - self.measured_pose) < EPSILON

    def move(self):
        if self.noise:
            vel_noise = np.random.normal(0, self.noise, 2)
            pos_noise = np.random.normal(0, self.noise, 2)
            vel_measure_noise = np.random.normal(0, self.noise, 2)
            self.velocity = self.target_velocity + vel_noise
            self.measured_velocity = self.velocity + vel_measure_noise
            self._ground_pose += self.dt * self.velocity
            self.measured_pose = self._ground_pose + pos_noise
        else:
            self.velocity = self.target_velocity
            self.measured_velocity = self.velocity
            self._ground_pose += self.dt * self.velocity
            self.measured_pose = self._ground_pose

    def choose_preferred_velocity(self):
        offset_to_goal = self._goal - self.measured_pose
        dist_to_goal = np.linalg.norm(offset_to_goal)

        if self.preferred_speed * self.dt > dist_to_goal:
            # Can reach goal in one step
            self.preferred_velocity = offset_to_goal / self.dt
        else:
            # Want to move at preferred speed in unit direction of goal
            self.preferred_velocity = self.preferred_speed * \
                                      offset_to_goal / dist_to_goal

    @abstractmethod
    def choose_target_velocity(self, neighbours):
        pass
