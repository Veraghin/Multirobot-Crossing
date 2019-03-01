from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
  data = np.genfromtxt('/tmp/gazebo_exercise.txt', delimiter=',')

  plt.figure()
  
  num_bots = int(len(data[0])/2)
  for x in range(num_bots):
    plt.plot(data[:, 2*x], data[:, (2*x)+1], 'g', label=str(x))

  plt.plot([-2, 2], [-2, -2], 'k')
  plt.plot([-2, 2], [2, 2], 'k')
  plt.plot([-2, -2], [-2, 2], 'k')
  plt.plot([2, 2], [-2, 2], 'k')
  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-2.5, 2.5])
  plt.ylim([-2.5, 2.5])

  if data.shape[1] == 6:
    plt.figure()
    error = np.linalg.norm(data[:, :2] - data[:, 3:5], axis=1)
    plt.plot(error, c='b', lw=2)
    plt.ylabel('Error [m]')
    plt.xlabel('Timestep')

  plt.show()
