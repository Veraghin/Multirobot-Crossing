import math
import os
import sys
import argparse
import numpy as np

# This file writes launch file only

def run(args):

  num_bots = args.num
  
  filename = "launch/gazebo_simple_"+str(num_bots)+".launch"
  file = open(filename, "w+")
  
  
  file.write('<launch>\n  <!--\n     Default GUI to true for local Gazebo client. Allows override\n       to set disable GUI for RoboMaker. See part0.launch.\n  -->\n  <arg name="use_gui" default="true"/>\n  <arg name="model" default="burger" doc="model type [burger, waffle, waffle_pi]"/>\n  <include file="$(find gazebo_ros)/launch/empty_world.launch">\n    <arg name="world_name" value="$(find exercises)/Multirobot-Crossing/worlds/simple.world"/>\n    <arg name="paused" value="true"/>\n    <arg name="use_sim_time" value="true"/>\n    <arg name="gui" value="$(arg use_gui)"/>\n    <arg name="headless" value="false"/>\n    <arg name="debug" value="false"/>\n  </include>\n\n')
  
  
  
  for i in range(num_bots):
    theta = 2 * np.pi * i / num_bots
    pos = 2.5 * np.array([np.cos(theta), np.sin(theta)])
    
    direction = -pos
    angle = np.arctan2(direction[1], direction[0])
    
    file.write('  <group ns ="r'+str(i)+'">\n')
    file.write('    <param name="robot_description" command="$(find xacro)/xacro --inorder $(find turtlebot3_description)/urdf/turtlebot3_$(arg model).urdf.xacro" />\n')
    file.write('    <node pkg="gazebo_ros" type="spawn_model" name="spawn_urdf"  args="-urdf -model turtlebot3_'+str(i)+'$(arg model) -x '+str(pos[0])+' -y '+str(pos[1])+' -z 0.0 -Y '+str(angle)+' -param robot_description" />\n')  
    file.write('  </group>\n')
    
  file.write('</launch>')
    
  
  
   
if __name__ == '__main__':
  # Assign mode:=malicious as arg to controller launch file to run malicious version
  #        num:=4 defines 4 robots (Only works for 4 currently)
  parser = argparse.ArgumentParser(description='Runs HRVO navigation')
  parser.add_argument('--num', type=int)
  args, unknown = parser.parse_known_args()
  run(args)
