import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.animation import FuncAnimation

from agents.CooperativeAgent import CooperativeAgent
from agents.SelfishAgent import SelfishAgent

NUM_ROBOTS = 20


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
starting_positions = []
theta = 2 * np.pi / NUM_ROBOTS
for j in range(20):
    starting_positions.append(np.array([18 * np.cos(theta * j), 18 * np.sin(theta * j)]))

for i in range(NUM_ROBOTS-1):
    ln, _ = plt.plot([], [], cs[i], animated=True)
    agent = CooperativeAgent(i, starting_positions[i], -starting_positions[i])

    robots_lines.append(ln)
    robot_location_hist.append([np.copy(starting_positions[i])])
    robots.append(agent)

i = NUM_ROBOTS - 1
ln, _ = plt.plot([], [], cs[i], animated=True)
agent = SelfishAgent(i, starting_positions[i], -starting_positions[i])
robots_lines.append(ln)
robot_location_hist.append([np.copy(starting_positions[i])])
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
