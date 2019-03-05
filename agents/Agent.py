#from abc import ABC, abstractmethod
import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

X = 0
Y = 1
YAW = 2


class Agent(object):
    def __init__(self, number, publisher, position, goal, orientation,
                 name='turtlebot3_', holonomic=False, radius=0.105,
                 wheel_base=0.016, time_to_orientation=0.2, simulation_step=0.1,
                 preferred_speed=0.15, max_speed=0.22, neighbour_range=np.infty,
                 noise=0.0):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
        self._number = number
        self._publisher = publisher
        self._id = name + str(number) + 'burger'
        self._ground_pose = position
        self.measured_pose = position
        self._goal = goal
        self.target_velocity = np.zeros(2, dtype=np.float32)
        self._ground_velocity = np.zeros(2, dtype=np.float32)
        self.measured_velocity = np.zeros(2, dtype=np.float32)
        self.dt = simulation_step
        self.noise = noise
        self.radius = radius
        self.preferred_speed = preferred_speed
        self.max_speed = max_speed
        self.preferred_velocity = np.zeros(2, dtype=np.float32)
        self.holonomic = holonomic
        self.neighbour_range = neighbour_range
        if not holonomic:
            self._ground_orientation = orientation
            self.measured_orientation = orientation
            self.wheel_base = wheel_base
            self.time_to_orientation = time_to_orientation

    # Returns advertised position; by default is robot's current knowledge of
    # its position
    def get_position(self):
        return self.measured_pose

    # Returns advertised velocity; by default is robot's current knowledge of
    # its velocity
    def get_velocity(self):
        return self.measured_velocity
        
    # Returns current orientation
    def get_orientation(self):
        return self.measured_orientation
        
    @property
    def publisher(self):
      return self._publisher
    
    @property
    def number(self):
      return self._number
      
    @property
    def ready(self):
      return not np.isnan(self.get_position()[0])

    # Boolean test for being within threshold distance of goal
    def at_goal(self):
        return np.linalg.norm(self._goal - self.measured_pose) < \
               self.radius * 0.3 + self.noise

    # Updates agent position based on gazebo location readings 
    def callback(self, msg):
      idx = [i for i, n in enumerate(msg.name) if n == self._id]
      if not idx:
        raise ValueError('Specified name "{}" does not exist.'.format(self._id))
      idx = idx[0]
      self._ground_pose[X] = msg.pose[idx].position.x
      self._ground_pose[Y] = msg.pose[idx].position.y
      _, _, yaw = euler_from_quaternion([
        msg.pose[idx].orientation.x,
        msg.pose[idx].orientation.y,
        msg.pose[idx].orientation.z,
        msg.pose[idx].orientation.w])
      self._ground_orientation = yaw

    def compute_wheel_speeds(self, target_velocity):
        orientation = self.get_orientation()
        target_orientation = orientation if self.at_goal() else np.arctan2(
            target_velocity[1], target_velocity[0])

        orientation_diff = np.fmod(target_orientation - orientation, 2 * np.pi)

        if orientation_diff < -np.pi:
            orientation_diff += 2 * np.pi
        if orientation_diff > np.pi:
            orientation_diff -= 2 * np.pi

        speed_diff = (orientation_diff * self.wheel_base) \
            / self.time_to_orientation

        if speed_diff > 2 * self.max_speed:
            speed_diff = 2 * self.max_speed
        elif speed_diff < -2 * self.max_speed:
            speed_diff = -2 * self.max_speed

        target_speed = np.linalg.norm(target_velocity)

        if target_speed + 0.5 * abs(speed_diff) > self.max_speed:
            if speed_diff >= 0:
                right_wheel = self.max_speed
                left_wheel = self.max_speed - speed_diff
            else:
                left_wheel = self.max_speed
                right_wheel = self.max_speed + speed_diff
        elif target_speed - 0.5 * abs(speed_diff) < -self.max_speed:
            if speed_diff >= 0:
                left_wheel = -self.max_speed
                right_wheel = speed_diff - self.max_speed
            else:
                right_wheel = -self.max_speed
                left_wheel = -self.max_speed - speed_diff
        else:
            right_wheel = target_speed + 0.5 * speed_diff
            left_wheel = target_speed - 0.5 * speed_diff
        return np.array([left_wheel, right_wheel])

    # Updates the robot's position according to its target_velocity
    def move(self):
        if self.noise:
            vel_noise = np.random.normal(0, self.noise, 2)
            pos_noise = np.random.normal(0, self.noise, 2)
            vel_measure_noise = np.random.normal(0, self.noise, 2)
            orientation_noise = np.random.normal(0, self.noise)
            orientation_measured_noise = np.random.normal(0, self.noise)
        else:
            vel_noise = 0
            pos_noise = 0
            vel_measure_noise = 0
            orientation_noise = 0
            orientation_measured_noise = 0

        if self.holonomic:
            self._ground_velocity = self.target_velocity + vel_noise
            self.measured_velocity = self._ground_velocity + vel_measure_noise
            self._ground_pose += self.dt * self._ground_velocity
            self.measured_pose = self._ground_pose + pos_noise
        else:
            average_wheel_speed = np.mean(self.target_velocity)
            wheel_speed_diff = self.target_velocity[1] - self.target_velocity[0]
            orientation = self.get_orientation()

            delta_pose = average_wheel_speed * np.array([
                np.cos(orientation), np.sin(orientation)]) + vel_noise
            self._ground_pose += self.dt * delta_pose
            self.measured_pose = self._ground_pose + pos_noise

            self.orientation_change = wheel_speed_diff * self.dt / \
                self.wheel_base + orientation_noise + orientation_measured_noise
                
            self._ground_orientation += wheel_speed_diff * self.dt / \
                self.wheel_base + orientation_noise
            self.measured_orientation = self._ground_orientation + \
                orientation_measured_noise

            self._ground_velocity = average_wheel_speed * np.array([
                np.cos(self.measured_orientation),
                np.sin(self.measured_orientation)
            ]) + vel_noise
            self.measured_velocity = self._ground_velocity + vel_measure_noise
            
      

    # Controller for deciding the target_velocity for this time step

    def choose_target_velocity(self, neighbours):
        pass
