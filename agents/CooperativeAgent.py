from Agent import Agent
from agent_utils import det, normal
from typing import List
import numpy as np
import heapq


# A velocity obstacle
class VelocityObstacle:
    def __init__(self, apex, side1, side2):
        self.apex = apex
        self.side1 = side1
        self.side2 = side2


# A candidate point that could be our target point
class Candidate:
    def __init__(self, position, vo1, vo2):
        self.position = position
        # The two indices of velocity objects that decide on the point
        self.vo1 = vo1
        self.vo2 = vo2


class CooperativeAgent(Agent):
    def choose_target_velocity(self, neighbours: List[Agent]):
        position = self.get_position()
        velocity = self.get_velocity()
        preferred_velocity = self.preferred_velocity

        velocity_obstacles = []
        sorted_neighbours = neighbours[:]
        sorted_neighbours.sort(
            key=lambda o: np.linalg.norm(o.get_position() - position))

        # Compute velocity obstacles for each neighbouring robot
        for neighbour in sorted_neighbours:
            neighbour_position = neighbour.get_position()
            neighbour_velocity = neighbour.get_velocity()
            offset = neighbour_position - position
            distance = np.linalg.norm(offset)

            if distance > 2 * self.radius:
                # Robots have not collided
                angle = np.arctan2(offset[1], offset[0])
                opening_angle = np.arcsin(2 * self.radius) / abs(offset)

                side1 = np.array([
                    np.cos(angle - opening_angle),
                    np.sin(angle - opening_angle)
                ])
                side2 = np.array([
                    np.cos(angle + opening_angle),
                    np.sin(angle + opening_angle)
                ])

                d = 2 * np.sin(opening_angle) * np.cos(opening_angle)

                if det(offset, preferred_velocity -
                        neighbour.preferred_velocity) > 0:
                    s = 0.5 * det(velocity - neighbour_velocity, side2) / d
                    apex = neighbour_velocity + s * side1 - (
                                self.noise * abs(offset) / (
                                    2 * self.radius)) * offset / distance
                else:
                    s = 0.5 * det(velocity - neighbour_velocity, side1) / d
                    apex = neighbour_velocity + s * side2 - (
                            self.noise * abs(offset) / (
                             2 * self.radius)) * offset / distance
            else:
                # Robots have collided
                apex = 0.5 * (neighbour_velocity + velocity) - (
                            self.noise + 0.5 * (2 * self.radius - abs(
                             offset) / self.dt)) * offset / distance
                side1 = normal(position, neighbour_position)
                side2 = -side1

            velocity_obstacles.append(VelocityObstacle(apex, side1, side2))

        # Create the set of all possible candidate points that we can choose to
        # move towards using the ClearPath algorithm

        candidates = []

        if np.linalg.norm(preferred_velocity) < self.max_speed:
            pos = preferred_velocity
        else:
            pos = self.max_speed * preferred_velocity / np.linalg.norm(
                preferred_velocity)

        heapq.heappush(candidates, (
            np.linalg.norm(preferred_velocity - pos),
            Candidate(pos, -1, -1)))

        for i, vo in enumerate(velocity_obstacles):
            dot_prod_1 = np.dot(preferred_velocity - vo.apex, vo.side1)
            dot_prod_2 = np.dot(preferred_velocity - vo.apex, vo.side2)

            if dot_prod_1 > 0 and det(vo.side1,
                                      preferred_velocity - vo.apex) > 1:
                pos = vo.apex + dot_prod_1 * vo.side1

                if np.linalg.norm(pos) < self.max_speed:
                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, i, i)))

            if dot_prod_2 > 0 and det(vo.side2,
                                      preferred_velocity - vo.apex) > 1:
                pos = vo.apex + dot_prod_2 * vo.side2

                if np.linalg.norm(pos) < self.max_speed:
                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, i, i)))

        for j, vo in enumerate(velocity_obstacles):
            discriminant = self.max_speed ** 2 - det(vo.apex, vo.side1) ** 2

            if discriminant > 0:
                t1 = -(vo.apex * vo.side1) + np.sqrt(discriminant)
                t2 = -(vo.apex * vo.side1) - np.sqrt(discriminant)

                if t1 >= 0:
                    pos = vo.apex + t1 * vo.side1

                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

                if t2 >= 0:
                    pos = vo.apex + t2 * vo.side1

                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

            discriminant = self.max_speed ** 2 - det(vo.apex, vo.side2) ** 2

            if discriminant > 0:
                t1 = -(vo.apex * vo.side2) + np.sqrt(discriminant)
                t2 = -(vo.apex * vo.side2) - np.sqrt(discriminant)

                if t1 >= 0:
                    pos = vo.apex + t1 * vo.side2

                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

                if t2 >= 0:
                    pos = vo.apex + t2 * vo.side2

                    heapq.heappush(candidates, (
                        np.linalg.norm(preferred_velocity - pos),
                        Candidate(pos, -1, j)))

        for i in range(len(velocity_obstacles)):
            for j in range(i + 1, len(velocity_obstacles)):
                vo1 = velocity_obstacles[i]
                vo2 = velocity_obstacles[j]

                d = det(vo1.side1, vo2.side1)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.side1) / d
                    t = det(vo2.apex - vo1.apex, vo1.side1) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.side1

                        if np.linalg.norm(pos) < self.max_speed:
                            heapq.heappush(candidates, (
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.side2, vo2.side1)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.side1) / d
                    t = det(vo2.apex - vo1.apex, vo1.side2) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.side2

                        if np.linalg.norm(pos) < self.max_speed:
                            heapq.heappush(candidates, (
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.side1, vo2.side2)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.side2) / d
                    t = det(vo2.apex - vo1.apex, vo1.side1) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.side1

                        if np.linalg.norm(pos) < self.max_speed:
                            heapq.heappush(candidates, (
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

                d = det(vo1.side2, vo2.side2)

                if d != 0:
                    s = det(vo2.apex - vo1.apex, vo2.side2) / d
                    t = det(vo2.apex - vo1.apex, vo1.side2) / d

                    if s >= 0 and t >= 0:
                        pos = vo1.apex + s * vo1.side2

                        if np.linalg.norm(pos) < self.max_speed:
                            heapq.heappush(candidates, (
                                np.linalg.norm(preferred_velocity - pos),
                                Candidate(pos, i, j)))

        # Choose the best candidate point that is closest to the desired
        # velocity towards the goal but will not cause any collisions or if no
        # such point exists, the point that will not collide with the largest
        # n closest robots
        optimal = -1
        for _ in range(len(candidates)):
            _, candidate = heapq.heappop(candidates)

            valid = True

            for i, vo in enumerate(velocity_obstacles):
                if i != candidate.vo1 and i != candidate.vo2 and \
                        det(vo.side2, candidate.position - vo.apex) < 0 < \
                        det(vo.side1, candidate.position - vo.apex):
                    valid = False

                    if i > optimal:
                        optimal = i
                        self.target_velocity = candidate.position

                    break

            if valid:
                self.target_velocity = candidate.position
                break
