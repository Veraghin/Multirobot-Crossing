import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from simulator import Simulator

modes = ["coop", "unaware", "aware"]
noise_values = [0.01, 0.03, 0.05]
num_robots_for_noise = [2, 4, 8]
noise_output = [[[[] for m in modes] for n in num_robots_for_noise] for n in noise_values]
for repetition in range(5):
    start = time.time()
    for n in range(len(noise_values)):
        for i in range(len(num_robots_for_noise)):
            for mode in range(len(modes)):
                sim = Simulator(num_robots_for_noise[i])
                steps, collisions = sim.run(mode=modes[mode], current_noise=noise_values[n])
                noise_output[n][i][mode].append((steps, collisions))
    end = time.time()
    print("Step %d done, took %f" % (repetition, (end - start)))

# Plot of min, median and max time taken to get to the goal,
# divided up by noise in the system and mode of operation of the robots
fig = plt.plot(1)
for n in range(len(noise_values)):
    for m in range(len(modes)):
        mine = [[] for i in num_robots_for_noise]
        med = [[] for i in num_robots_for_noise]
        maxe = [[] for i in num_robots_for_noise]
        for i in range(len(num_robots_for_noise)):
            val = noise_output[n][i][m]
            mine[i].append(np.amin(val, axis=0)[0])
            med[i].append(np.median(val, axis=0)[0])
            maxe[i].append(np.amax(val, axis=0)[0])
        plt.subplot(3, 3, 3 * n + m + 1)
        plt.xtics
        plt.plot(num_robots_for_noise, mine, 'rv', num_robots_for_noise, med, 'bo', num_robots_for_noise, maxe, 'g^')
plt.show()
# Collisions against number of robots, with malicious, aware-oracle and aware-history subplots
noise_values = [0.01, 0.03, 0.05]
num_robots_for_collision = [2, 4, 8]
collision_modes = ["unaware", "aware-oracle", "aware-history"]
collision_output = [[[[] for m in modes] for n in num_robots_for_noise] for n in noise_values]
for repetition in range(5):
    start = time.time()
    for n in range(len(noise_values)):
        for i in range(len(num_robots_for_collision)):
            for mode in range(len(collision_modes)):
                sim = Simulator(num_robots_for_collision[i])
                if collision_modes[mode] == "unaware":
                    steps, collisions = sim.run(mode="coop", current_noise=noise_values[n])
                elif collision_modes[mode] == "aware-oracle":
                    steps, collisions = sim.run(mode="aware", current_noise=noise_values[n], identifier="oracle")
                else:
                    steps, collisions = sim.run(mode="aware", current_noise=noise_values[n], identifier="history")
                collision_output[n][i][mode].append((steps, collisions))
    end = time.time()
    print("Step %d done, took %f" % (repetition, (end - start)))

# Plot of min, median and max time taken to get to the goal,
# divided up by noise in the system and mode of operation of the robots
fig, axes = plt.subplots(3, 3)
outsideX = ["Unaware", "Aware\nOracle", "Aware\nHistory"]
outsideY = ["Noise: %.2f" % n for n in noise_values]
pad = 5  # in points
for ax, col in zip(axes[0], outsideY):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:, 0], outsideX):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
plt.setp(axes.flat, xlabel='X-label', ylabel='Y-label')
for n in range(len(noise_values)):
    for m in range(len(collision_modes)):
        mine = [[] for i in num_robots_for_collision]
        med = [[] for i in num_robots_for_collision]
        maxe = [[] for i in num_robots_for_collision]
        for i in range(len(num_robots_for_collision)):
            val = collision_output[n][m][i]
            mine[i].append(np.amin(val, axis=0)[1])
            med[i].append(np.median(val, axis=0)[1])
            maxe[i].append(np.amax(val, axis=0)[1])
        axes[n][m].set_xticks(num_robots_for_collision)
        axes[n][m].plot(num_robots_for_collision, mine, 'rv', num_robots_for_collision, med, 'bo',
                        num_robots_for_collision, maxe, 'g^')
fig.tight_layout()
fig.subplots_adjust(left=0.2, top=0.95)
plt.show()
