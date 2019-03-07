#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys
from agents.Agent import Agent
from agents.CooperativeAgent import CooperativeAgent
from agents.SuspiciousAgent import SuspiciousAgent
from agents.SelfishAgent import SelfishAgent
from agents.malicious_detectors import history

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# For robot information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
import math

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)

def run(args):
  rospy.init_node('hrvo_navigation')
  number_of_bots = int(args.num)
  mode = args.mode
  
  # Update control every 100 ms.
  rate_limiter = rospy.Rate(100)
  robot_list = []
  pose_history = []
  
  # Initialise Agents
  for i in range(number_of_bots):
  
    theta = 2 * np.pi * i / number_of_bots
    pos = 2.5 * np.array([np.cos(theta), np.sin(theta)])

    direction = -pos
    orientation = np.arctan2(direction[1], direction[0])
  
    publisher = rospy.Publisher('/r'+str(i)+'/cmd_vel', Twist, queue_size=5)
    
    if mode == 'malicious':
      if i == 1:
          agent = SelfishAgent(i, publisher, pos, -pos, orientation)
      else:
          agent = SuspiciousAgent(i, publisher, pos, -pos,
                       orientation, malicious_identifier=history(0.085))
    else:
      agent = CooperativeAgent(i, publisher, pos, -pos, orientation)
      
    robot_list.append(agent)
    
    vel_msg = Twist()
    vel_msg.angular.z = agent._ground_orientation
    agent.publisher.publish(vel_msg)
    
  # Opening log file
  with open('/tmp/gazebo_exercise.txt', 'w'):
    pass

  while not rospy.is_shutdown():  
    pos_list = []
    
    # Let all robots choose where they want to move first
    for robot in robot_list:
      robot.choose_target_velocity([r for r in robot_list if r._id != robot._id])

    # Make sure all measurements are ready.
    for robot in robot_list:
      if not robot.ready:
        rate_limiter.sleep()
        continue

      robot.move()

      v = robot.get_velocity()
      w = robot.get_orientation()
      
      # Publishing next step velocity / yaw to robot
      vel_msg = Twist()
      vel_msg.linear.x = v[0]*np.cos(w) + v[1]*np.sin(w)
      vel_msg.angular.z = robot.orientation_change
      robot.publisher.publish(vel_msg)
      pos_list.append(robot.get_position())

    print(np.concatenate(pos_list))
    
    # Log robot positions in /tmp/gazebo_exercise.txt
    pose_history.append(np.concatenate(pos_list, axis=0))
    if len(pose_history) % 10:
      with open('/tmp/gazebo_exercise.txt', 'a') as fp:
        fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
        pose_history = []
    rate_limiter.sleep()
    print()


if __name__ == '__main__':
  # Assign mode:=malicious as arg to controller launch file to run malicious version
  #        num:=4 defines 4 robots (Only works for 4 currently)
  parser = argparse.ArgumentParser(description='Runs potential field navigation')
  parser.add_argument('--num', type=int)
  parser.add_argument('--mode', type=str)
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
