#!/usr/bin/env python3

# Import neccesary modules
# Set matplotlib to inline mode for the Jupyter notebook visualisations

# %matplotlib inline

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid

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
    
# Initiate ROS node
rospy.init_node('localization')

# Wait until the node exists else it will throw an error
rospy.wait_for_service('static_map')

# Get the map
recMap = getMap()

# Deserialise into a 2D array
mapData = np.split(np.array(recMap.data), recMap.info.height)

# Visualize the map into the console
for row in mapData:
    rowStr = ""
    for col in row:
        if col == 0:
            rowStr += " "
        elif col == 100:
            rowStr += "#"
        else:
            rowStr += "?"
    print(rowStr)
