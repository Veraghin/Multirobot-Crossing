import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.animation import FuncAnimation

from agents.SuspiciousAgent import SuspiciousAgent
from agents.SelfishAgent import SelfishAgent
from agents.malicious_detectors import history

NUM_ROBOTS = 10


def colors_from(cmap_name, ncolors):
    cm = plt.get_cmap(cmap_name)
    cm_norm = colors.Normalize(vmin=0, vmax=ncolors - 1)
    scalar_map = plt.cm.ScalarMappable(norm=cm_norm, cmap=cm)
    return [scalar_map.to_rgba(i) for i in range(ncolors)]


# Setup plots
cs = colors_from("jet", NUM_ROBOTS)
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

# Initialise robots with starting positions and goals
robots_lines = []
robot_location_hist = []
robots = []
for i in range(NUM_ROBOTS):
    theta = 2 * np.pi * i / NUM_ROBOTS
    pos = 15 * np.array([np.cos(theta), np.sin(theta)])
    ln, _ = plt.plot([], [], cs[i], animated=True)
    if i == 1:
        agent = SelfishAgent(i, pos, -pos, max_speed=4., preferred_speed=2.,
                             radius=1)
    else:
        agent = SuspiciousAgent(i, pos, -pos, max_speed=4., preferred_speed=2.,
                                radius=1, malicious_identifier=history(0.085))

    robots_lines.append(ln)
    robot_location_hist.append([np.copy(pos)])
    robots.append(agent)


# Main simulation loop
def update(_):
    # If all robots are at goal, exit
    at_goals = True
    for robot in robots:
        if not robot.at_goal():
            at_goals = False
    if at_goals:
        quit()

    # Let all robots choose where they want to move first
    for robot in robots:
        robot.choose_target_velocity([r for r in robots if r.id != robot.id])

    # Once all robots decided, they all move at the same time and update the
    # plots
    for robot in range(len(robots_lines)):
        robots[robot].move()
        robot_location_hist[robot].append(np.copy(robots[robot]._ground_pose))
        xs, ys = zip(*robot_location_hist[robot])
        robots_lines[robot].set_data(xs, ys)

    return robots_lines


_ = FuncAnimation(fig, update, init_func=(lambda: robots_lines),
                  interval=100, blit=True)
plt.show()
