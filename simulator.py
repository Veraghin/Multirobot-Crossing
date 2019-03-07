import numpy as np
import argparse
import time
from agents.CooperativeAgent import CooperativeAgent
from agents.SelfishAgent import SelfishAgent
from agents.SuspiciousAgent import SuspiciousAgent
from agents.malicious_detectors import history, oracle


class Simulator:
    def __init__(self, total_num_robots, radius=10):
        self.num_robots = int(total_num_robots)
        self.starting_radius = radius
        self.robot_rad = .5
        self.pref_speed = 2.0
        self.max_speed = 4.0

    def run(self, mode="coop", identifier="oracle", detection_range=np.infty, current_noise=0.0):
        if mode not in ["coop", "unaware", "aware"]:
            mode = "coop"
        # Initialise robots with starting positions and goals
        robot_location_hist = []
        robots = []
        for i in range(self.num_robots):
            theta = 2 * np.pi * i / self.num_robots
            pos = self.starting_radius * np.array([np.cos(theta), np.sin(theta)])
            diff = (-pos) - pos
            orientation = np.arctan2(diff[1], diff[0])
            if i == self.num_robots - 1 and mode != "coop":
                agent = SelfishAgent(i, None, pos, -pos, orientation, radius=self.robot_rad,
                                     preferred_speed=self.pref_speed,
                                     max_speed=self.max_speed,
                                     neighbour_range=detection_range, noise=current_noise)
            elif mode == "coop" or mode == "unaware":
                agent = CooperativeAgent(i, None, pos, -pos, orientation, radius=self.robot_rad,
                                         preferred_speed=self.pref_speed,
                                         max_speed=self.max_speed,
                                         neighbour_range=detection_range, noise=current_noise)
            else:  # mode == "aware"
                if identifier == "oracle":
                    agent = SuspiciousAgent(i, None, pos, -pos, orientation, radius=self.robot_rad,
                                            preferred_speed=self.pref_speed,
                                            max_speed=self.max_speed,
                                            neighbour_range=detection_range, noise=current_noise,
                                            malicious_identifier=oracle(self.num_robots - 1))
                else:
                    agent = SuspiciousAgent(i, None, pos, -pos, orientation, radius=self.robot_rad,
                                            preferred_speed=self.pref_speed,
                                            max_speed=self.max_speed,
                                            neighbour_range=detection_range, noise=current_noise,
                                            malicious_identifier=history(0.085))
            robot_location_hist.append([np.copy(pos)])
            robots.append(agent)

        def collisions(robots):
            num = 0
            for r in robots:
                others = [o for o in robots if r.number != o.number]
                for o in others:
                    # radius=0.105
                    if np.linalg.norm(r._ground_pose - o._ground_pose) <= 2 * self.robot_rad:
                        num += 1
            return num

        # Run simulation
        steps_taken = 0
        collisions_total = 0
        while True:
            # If all robots are at goal, exit
            at_goals = True
            for robot in robots:
                if not robot.at_goal():
                    at_goals = False
            if at_goals:
                break
            else:
                steps_taken += 1
            # Let all robots choose where they want to move first
            for robot in robots:
                robot.choose_target_velocity([r for r in robots if r.number != robot.number])
            # Once all robots decided, they all move at the same time and update the
            # plots
            for robot in range(len(robots)):
                robots[robot].move()
                robot_location_hist[robot].append(np.copy(robots[robot]._ground_pose))
            collisions_total += collisions(robots)
        return steps_taken, collisions_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs the simulation simulation')
    parser.add_argument('--number_of_robots', action='store', type=int, default=4,
                        help='The number of robots in the simulation')
    args = parser.parse_args()
    sim = Simulator(args.number_of_robots)
    start = time.time()
    print(sim.run(mode="aware", current_noise=0.0))
    end = time.time()
    print(end - start)
