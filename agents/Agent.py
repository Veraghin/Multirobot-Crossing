from abc import ABC, abstractmethod
import numpy as np


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
        self.max_speed = max_speed
        self.preferred_velocity = np.zeros(2, dtype=np.float32)

    # Returns advertised position; by default is robot's current knowledge of
    # its position
    def get_position(self):
        return self.measured_pose

    # Returns advertised velocity; by default is robot's current knowledge of
    # its velocity
    def get_velocity(self):
        return self.measured_velocity

    # Boolean test for being within threshold distance of goal
    def at_goal(self):
        return np.linalg.norm(self._goal - self.measured_pose) < \
               self.radius * 0.1 + self.noise

    # Updates the robot's position according to its target_velocity
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

    # Controller for deciding the target_velocity for this time step
    @abstractmethod
    def choose_target_velocity(self, neighbours):
        pass
