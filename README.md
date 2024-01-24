# Maze Escape Robot Documentation
## Overview
The goal of this project is to develop ROS scripts in order to make an "autonomous" maze-solving robot. The robot is equipped with a laser scanner and utilizes a combination of map processing, machine learning-based localization, and search algorithms to navigate through the maze and reach the exit.

## Project Structure
The project consists of the following main files:

- mazeSolver.py: The main Python script responsible for solving the maze. It uses ROS messages for laser scan data, map information, and pose data. The script performs map processing, k-nearest neighbors (kNN) localization, depth-first search (DFS) for pathfinding, and communicates with the move_base module for controlling the robot's movement.

- mapVisualisation.py: A Python script containing classes for visualizing different aspects of the map and the robot's movements. It uses Matplotlib for creating visual representations of the map, laser scan data, and the robot's path.

- launchSimulation.launch: A ROS launch file for starting the Gazebo simulation environment, loading the map, spawning the robot, launching RViz for visualization, and executing the maze-solving script.

## Usage
- You need to get the Docker image containing all the frameworks and libraries needed to launch the project! https://github.com/TW-Robotics/Docker-ROS/tree/mre2_ear_ws2023
- matplotlib might be missing from the image therefore you might need to install it by entering this command before launching the project `python -m pip install -U matplotlib`
- There's a few constants you are able to modify in order to test different configuration for the project!
  - The main one is the position of the robot in the maze which can be modified in the launchSimulation.launch at line 8 and 9.
  - You can choose to show or not the plots drawn by matplotlib if you don't need to visualize the map. In the mazeSolver.py, change the `ALLOW_VISUALISATION = True` to `False` line 19 to stop the visualisation.
  - Also you're able to modify the goal (or exit) of the maze by modifying the coordinates of the constant `GOAL = [3, 3]` line 21 on the mazeSolver.py.
  - Finally, you can change the time between the publishing of the movements for the robot line 21! `TIME_BETWEEN_PUBLISH = 10` You might want to avoid putting the value too low...
- Once everything is set, you're can launch the launchSimulation.launch file with `roslaunch soar_maze_escape launchSimulation.launch`

## How it works

### Map Processing
The mazeSolver.py script processes the map using the map service and laser scan data. It extracts wall and free positions, aligns them to the grid, and uses kNN for localization. The map is represented as a 2D array of integers, where 0 represents a free position and 1 represents a wall, and is used all throughout the project.

### Localization
In order to localize the robot, the script uses a kNN model trained on the map data. The model is trained on a set of 2D coordinates (X) and their corresponding map values either being 100 or 0 for walls or free positions (y).  

### Pathfinding
The robot utilizes depth-first search (DFS) to find a path from its current position to the exit of the maze. The algorithm is implemented in a custom recursive function and outputs a list of waypoints.

### Visualization
The mapVisualisation.py script provides visualizations for map data, kNN model, graph generation, and the robot's path. These visualizations help in understanding the robot's perception of the environment and the execution of the pathfinding algorithm.
