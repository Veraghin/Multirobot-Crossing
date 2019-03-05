from agents.Agent import Agent
from agents.agent_utils import det, normal
import numpy as np


# A velocity obstacle
class VelocityObstacle:
    def __init__(self, apex, left, right):
        self.apex = apex
        self.left = left
        self.right = right


# A candidate point that could be our target point
class Candidate:
    def __init__(self, position, vo1, vo2):
        self.position = position
        # The two indices of velocity objects that decide on the point
        self.vo1 = vo1
        self.vo2 = vo2


class CooperativeAgent(Agent):
    # Controller for the process of choosing the target velocity for the next
    # time step
    def choose_target_velocity(self, neighbours):
        if self.at_goal():
            self.target_velocity = np.zeros(2, dtype=np.float32)
            return

        position = self.get_position()
        velocity = self.get_velocity()
        preferred_velocity = self.choose_preferred_velocity()

        # Compute velocity obstacles for each neighbouring robot
        velocity_obstacles = self.compute_hrvos(
            position, velocity, neighbours[:])

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
        target_velocity = self.choose_best_velocity(optimal_candidates)
        if self.holonomic:
            self.target_velocity = target_velocity
        else:
            self.target_velocity = self.compute_wheel_speeds(target_velocity)

    # Finds velocity in direction of goal at desired speed
    def choose_preferred_velocity(self):
        offset_to_goal = self._goal - self.measured_pose
        dist_to_goal = np.linalg.norm(offset_to_goal)

        if self.preferred_speed * self.dt > dist_to_goal:
            # Can reach goal in one step
            return offset_to_goal / self.dt
        else:
            # Want to move at preferred speed in unit direction of goal
            return self.preferred_speed * offset_to_goal / dist_to_goal

    # Computes all HRVOs created by the neighbouring robots
    def compute_hrvos(self, position, velocity, neighbours):
        velocity_obstacles = []
        neighbours.sort(
            key=lambda o: np.linalg.norm(o.get_position() - position))

        # Velocity obstacles from neighbours are added in order of increasing
        # distance so that more distant ones will be discarded first when we
        # cannot satisfy them all
        for neighbour in neighbours:
            neighbour_position = neighbour.get_position()
            neighbour_velocity = neighbour.get_velocity()
            offset = neighbour_position - position
            distance = np.linalg.norm(offset)
            if distance > self.neighbour_range:
                continue

            if distance > 2 * self.radius:
                # Robots have not collided
                angle = np.arctan2(offset[1], offset[0])
                opening_angle = np.arcsin(2 * self.radius / distance)

                left = np.array([
                    np.cos(angle + opening_angle),
                    np.sin(angle + opening_angle)
                ])
                right = np.array([
                    np.cos(angle - opening_angle),
                    np.sin(angle - opening_angle)
                ])

                d = 2 * np.sin(opening_angle) * np.cos(opening_angle)

                if det(offset, velocity - neighbour_velocity) > 0:
                    # vA on left of centre line
                    s = 0.5 * det(velocity - neighbour_velocity, left) / d
                    apex = np.array(neighbour_velocity + s * right - (
                            self.noise * abs(offset) / (
                             2 * self.radius)) * offset / distance)
                else:
                    # va on right of centre line
                    s = 0.5 * det(velocity - neighbour_velocity, right) / d
                    apex = np.array(neighbour_velocity + s * left - (
                            self.noise * abs(offset) / (
                             2 * self.radius)) * offset / distance)
            else:
                # Robots have collided
                apex = np.array(0.5 * (neighbour_velocity + velocity) - (
                        self.noise + 0.5 * (2 * self.radius - abs(
                         offset) / self.dt)) * offset / distance)
                right = normal(position, neighbour_position)
                left = -right
            velocity_obstacles.append(VelocityObstacle(apex, left, right))

        return velocity_obstacles

    # Projects preferred velocity onto HRVOs to produce a candidate point
    # Slight hack to only project onto left edge of HRVO to ensure agreement
    # on which is the best way to turn when turning left or right would be
    # equally good
    def compute_projection_candidates(self, velocity_obstacles,
                                      preferred_velocity):
        candidates = []
        for i, vo in enumerate(velocity_obstacles):
            dot_prod = np.dot(preferred_velocity - vo.apex, vo.left)

            if dot_prod > 0 > det(vo.left, preferred_velocity - vo.apex):
                pos = vo.apex + dot_prod * vo.left

                if np.linalg.norm(pos) < self.max_speed:
                    candidates.append((
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, i, i)))
        return candidates

    # Finds max velocity on HRVO by finding the point of intersection between
    # a circle around each agent with the HRVO
    def compute_max_candidates(self, velocity_obstacles, preferred_velocity):
        candidates = []
        for j, vo in enumerate(velocity_obstacles):
            discriminant = self.max_speed ** 2 - det(vo.apex, vo.right) ** 2

            if discriminant > 0:
                t1 = -(np.dot(vo.apex, vo.right)) + np.sqrt(discriminant)
                t2 = -(np.dot(vo.apex, vo.right)) - np.sqrt(discriminant)

                if t1 >= 0:
                    pos = vo.apex + t1 * vo.right

                    candidates.append((
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

                if t2 >= 0:
                    pos = vo.apex + t2 * vo.right

                    candidates.append((
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

            discriminant = self.max_speed ** 2 - det(vo.apex, vo.left) ** 2

            if discriminant > 0:
                t1 = -(np.dot(vo.apex, vo.left)) + np.sqrt(discriminant)
                t2 = -(np.dot(vo.apex, vo.left)) - np.sqrt(discriminant)

                if t1 >= 0:
                    pos = vo.apex + t1 * vo.left

                    candidates.append((
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

                if t2 >= 0:
                    pos = vo.apex + t2 * vo.left

                    candidates.append((
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))
        return candidates

    # Produces candidate points from the intersection of 2 HRVOs
    def compute_intersect_candidates(self, velocity_obstacles,
                                     preferred_velocity):
        candidates = []
        for i in range(len(velocity_obstacles)):
            for j in range(i + 1, len(velocity_obstacles)):
                vo1 = velocity_obstacles[i]
                vo2 = velocity_obstacles[j]

                d = det(vo1.right, vo2.right)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.right) / d
                    t = det(vo2.apex - vo1.apex, vo1.right) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.right

                        if np.linalg.norm(pos) < self.max_speed:
                            candidates.append((
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.left, vo2.right)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.right) / d
                    t = det(vo2.apex - vo1.apex, vo1.left) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.left

                        if np.linalg.norm(pos) < self.max_speed:
                            candidates.append((
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.right, vo2.left)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.left) / d
                    t = det(vo2.apex - vo1.apex, vo1.right) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.right

                        if np.linalg.norm(pos) < self.max_speed:
                            candidates.append((
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.left, vo2.left)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.left) / d
                    t = det(vo2.apex - vo1.apex, vo1.left) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.left

                        if np.linalg.norm(pos) < self.max_speed:
                            candidates.append((
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))
        return candidates

    # Finds the candidates that all avoid collisions with a maximum n closest
    # agents
    @staticmethod
    def find_optimal_candidates(candidates, velocity_obstacles):
        optimal_satisfied = -1
        optimal_candidates = []
        for d, candidate in candidates:
            valid = True

            i = 0
            for vo in velocity_obstacles:
                if i != candidate.vo1 and i != candidate.vo2 and \
                        det(vo.left, candidate.position - vo.apex) < 0 < \
                        det(vo.right, candidate.position - vo.apex):
                    valid = False

                    if i > optimal_satisfied:
                        optimal_satisfied = i
                        optimal_candidates = [(d, candidate)]
                    elif i == optimal_satisfied:
                        optimal_candidates.append((d, candidate))
                    i += 1

                    break
                i += 1

            if valid:
                if i > optimal_satisfied:
                    optimal_satisfied = i
                    optimal_candidates = [(d, candidate)]
                elif i == optimal_satisfied:
                    optimal_candidates.append((d, candidate))
        return optimal_candidates

    # Chooses the velocity closest to the preferred velocity (using insertion
    # order in the case of a tie)
    @staticmethod
    def choose_best_velocity(candidates):
        min_distance = np.infty
        target_velocity = None
        for d, candidate in candidates:
            if d < min_distance:
                min_distance = d
                target_velocity = candidate.position
        return target_velocity
