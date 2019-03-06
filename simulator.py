import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import argparse
from matplotlib.animation import FuncAnimation

from agents.CooperativeAgent import CooperativeAgent
from agents.SelfishAgent import SelfishAgent
from agents.SuspiciousAgent import SuspiciousAgent
from agents.malicious_detectors import history, oracle

class Simulator():
    def __init__(self,args):
        self.num_robots = args.number_of_robots
        self.starting_radius = 5

    def run(self, mode="coop", identifier="oracle", current_noise = 0.0):
        # Initialise robots with starting positions and goals
        self.steps_taken = 0
        robot_location_hist = []
        robots = []
        for i in range(self.num_robots):
            theta = 2 * np.pi * i / self.num_robots
            pos = self.starting_radius * np.array([np.cos(theta), np.sin(theta)])
            diff = (-pos) - pos
            orientation = np.arctan2(diff[1], diff[0])
            if i == self.num_robots - 1 and mode != "coop":
                agent = SelfishAgent(i, None, pos, -pos, orientation, noise=current_noise)
            elif mode == "coop":
                agent = CooperativeAgent(i, None, pos, -pos, orientation, noise=current_noise)
            elif mode == "aware":
                if identifier == "oracle":
                    agent = SuspiciousAgent(i, None, pos, -pos, orientation, noise=current_noise, malicious_identifier=oracle(self.num_robots -1))
                else:
                    agent = SuspiciousAgent(i, None, pos, -pos, orientation, noise=current_noise, malicious_identifier=history(0.085))
            robot_location_hist.append([np.copy(pos)])
            robots.append(agent)

        def collisions(robots):
            num = 0
            for r in robots:
                others = [o for o in robots if r.number != o.number]
                for o in others:
                    # radius=0.105
                    if np.linalg.norm(r._ground_pose - o._ground_pose) <= .21:
                        num +=1
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
    
    def output_to_csv(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs the simulation simulation')
    parser.add_argument('--number_of_robots', action='store', default=4, help='The number of robots in the simulation')
    args = parser.parse_args()
    sim = Simulator(args)
    print(sim.run())
    print(sim.run(mode="aware"))