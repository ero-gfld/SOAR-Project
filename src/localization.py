#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid

import matplotlib.pyplot as plt

import numpy as np

import copy
import time
import math

# Helper method for retrieving the map
def getMap() -> OccupancyGrid:
    """ Loads map from map service """
    # Create service proxy
    get_map = rospy.ServiceProxy('static_map', GetMap)
    # Call service
    recMap = get_map()
    recMap = recMap.map
    # Return
    return recMap

# Helper method to show the map
def showMap(freePositions, wallPositions, scanPositions):
    """ Displays map """
    # Store colours matching UAS TW colour scheme as dict 
    colourScheme = {
        "darkblue": "#143049",
        "twblue": "#00649C",
        "lightblue": "#8DA3B3",
        "lightgrey": "#CBC0D5",
        "twgrey": "#72777A"
    }

    ## Visualise transformed maze and scans
    # Create single figure
    plt.rcParams['figure.figsize'] = [7, 7]
    fig, ax = plt.subplots()

    # Plot data as points (=scatterplot) and label accordingly. The colours are to look nice with UAS TW colours
    ax.scatter(scanPositions[:,1], scanPositions[:,0], c="r", alpha=0.8, label="Laserscan")
    ax.scatter(wallPositions[:,1], wallPositions[:,0], c=colourScheme["darkblue"], alpha=1.0, s=6**2, label="Walls")
    ax.scatter(freePositions[:,1], freePositions[:,0], c=colourScheme["twgrey"], alpha=0.08, s=6**2, label="Unobstructed Space")
    ax.scatter([0], [0], c=colourScheme["twblue"], s=15**2, label="Scan Center")

    # Set axes labels and figure title
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Map and Laserscan Data Transformed into World Coordinates")

    # Set grid to only plot each metre
    ax.set_xticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_yticks = [-1, 0, 1, 2, 3, 4 ]

    # Move grid behind points
    ax.set_axisbelow(True)
    ax.grid()

    # Add labels
    ax.legend()

    # Show plot
    plt.show()
    
# Initiate ROS node
rospy.init_node('localization')

# Wait until the node exists else it will throw an error
rospy.wait_for_service('static_map')

# Get the map
recMap = getMap()

# Get the map data
mapData = np.split(np.array(recMap.data), recMap.info.height)

# Get data into wallPosition and freePosition arrays
wallPositions = np.array([])
freePositions = np.array([])
for i in range(len(mapData)):
    for j in range(len(mapData[i])):
        x = i * recMap.info.resolution + (recMap.info.origin.position.x + recMap.info.resolution / 2)
        y = j * recMap.info.resolution + (recMap.info.origin.position.y + recMap.info.resolution / 2)
        if mapData[i][j] == 100:
            wallPositions = np.append(wallPositions, [x, y])
        elif mapData[i][j] == 0:
            freePositions = np.append(freePositions, [x, y])

# Reshape arrays to be 2D
wallPositions = np.reshape(wallPositions, (-1, 2))
freePositions = np.reshape(freePositions, (-1, 2))

# Get the laser scan data
scanData = rospy.wait_for_message('/scan', LaserScan)

# Use np.arange to get the angles of each measurement
angles = np.arange(scanData.angle_min, scanData.angle_max, scanData.angle_increment)

# Convert angle-distance pairs to Cartesian coordinates
scanPositions = np.array([])
for i in range(len(angles)):
    x = scanData.ranges[i]*math.sin(angles[i])
    y = scanData.ranges[i]*math.cos(angles[i])
    scanPositions = np.append(scanPositions, [x, y])

# The walls having thickness laserscan and walls won't match up perfectly. We move each point away from the laserscan by half the map's resolution.
scanPositions = scanPositions + (recMap.info.resolution / 2)

# Reshape array to be 2D
scanPositions = np.reshape(scanPositions, (-1, 2))

# Show the map of the scan
showMap(freePositions, wallPositions, scanPositions)