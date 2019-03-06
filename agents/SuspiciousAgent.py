from agents.CooperativeAgent import CooperativeAgent, VelocityObstacle
from agents.agent_utils import det, normal
import numpy as np


class SuspiciousAgent(CooperativeAgent):
    def __init__(self, *args, **kwargs):
        malicious_identifier = kwargs.pop("malicious_identifier", lambda _: [])
        super(SuspiciousAgent, self).__init__(*args, **kwargs)
        self.malicious_identifier = malicious_identifier

    def compute_hrvos(self, position, velocity, neighbours):
        velocity_obstacles = []
        neighbours.sort(
            key=lambda o: np.linalg.norm(o.get_position() - position))

        malicious_neighbours = self.malicious_identifier(neighbours)

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

                if neighbour in malicious_neighbours:
                    apex = neighbour_velocity
                else:
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
                if neighbour in malicious_neighbours:
                    apex = neighbour_velocity
                else:
                    apex = np.array(0.5 * (neighbour_velocity + velocity) - (
                            self.noise + 0.5 * (2 * self.radius - abs(
                             offset) / self.dt)) * offset / distance)
                right = normal(position, neighbour_position)
                left = -right
            velocity_obstacles.append(VelocityObstacle(apex, left, right))

        return velocity_obstacles
