#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import neighbors

import numpy as np

import re

import copy
import time
import math

# Define colour scheme
colourScheme = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "lightblue": "#8DA3B3",
    "lightgrey": "#CBC0D5",
    "twgrey": "#72777A"
}

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

def visualiseClf(clf, X):
    # Visualise fitted model using built-in visualiser
    plt.rcParams['figure.figsize'] = [5, 5]
    _, ax = plt.subplots()

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=ListedColormap(
            [
                colourScheme["twgrey"],
                colourScheme["darkblue"]
            ]
        ),
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel="X-Coordinate[m]",
        ylabel="Y-Coordinate[m]",
        shading="auto",
    )

    # Set title and show plot
    ax.set_title("Fitted kNN Model representing the Map")
    plt.show()

# Helper function for colours
# See https://matplotlib.org/stable/tutorials/colors/colormaps.html for more information
def createColourmapToWhite(rgb, reverse=False):
    """ Creates a colourmap that fades from rgb colour to white """
    # Unpack tuple and manually create fades with a resolution of 256
    r, g, b = rgb
    N = 256
    vals = np.ones((N, 4))
    # Distinguish between reverse and non-reverse and invert linspace accordingly
    if reverse:
        vals[:, 0] = np.linspace(1, r/N, N)
        vals[:, 1] = np.linspace(1, g/N, N)
        vals[:, 2] = np.linspace(1, b/N, N)
    else:
        vals[:, 0] = np.linspace(r/N, 1, N)
        vals[:, 1] = np.linspace(g/N, 1, N)
        vals[:, 2] = np.linspace(b/N, 1, N)
    return ListedColormap(vals)

def hexstring2rgb(colourstring):
    """ Converts hex colours in string form to rgb values """
    i = re.compile('#')
    colourstring = re.sub(i, '', colourstring)
    return tuple(int(colourstring[i:i+2], 16) for i in (0, 2, 4))

# Create colourmap that fades from white to UAS TW blue for heatmap
heatColourMap = createColourmapToWhite( hexstring2rgb( colourScheme["twblue"] ), reverse=True)

# ----------------------------------------------------------------
    
# Initiate ROS node
rospy.init_node('localization')

# ----------------------------------
# Step 1: Get map and laserscan data
# ----------------------------------

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
# showMap(freePositions, wallPositions, scanPositions)

# ---------------------------------
# Step 2: Localisation using kNN
# ---------------------------------

print("Start predicting...")

# Use KNN to find the nearest neighbour for each point in the scan
knn = neighbors.KNeighborsClassifier(n_neighbors=1)

# X will be all the coordinates of the map's points and Y will be the class of the point (0 for free, 1 for wall)
X = np.concatenate((wallPositions, freePositions))
Y = np.concatenate((np.ones(len(wallPositions)), np.zeros(len(freePositions))))

# Fit the model
clf = knn.fit(X, Y)

# Visualise the model
# visualiseClf(clf, X)

# Define all possible poses (positions where no walls and coordinates that are whole numbers)
poses = []
for x in range(int(recMap.info.origin.position.x), int(recMap.info.origin.position.x + recMap.info.width * recMap.info.resolution) + 1, 1):
    for y in range(int(recMap.info.origin.position.y), int(recMap.info.origin.position.y + recMap.info.height * recMap.info.resolution) + 1, 1):
        if [x, y] not in wallPositions.tolist():
            poses.append([x, y])

# Iterate over all poses
allPredictions = {}
for pose in poses:
    # Transform the laserscan to the current pose
    transformedScan = scanPositions + pose
    
    # Predict the values using the fitted model
    predictions = clf.predict(transformedScan)
    
    # Count the number of "wall" predictions
    wallCount = np.count_nonzero(predictions)

    # Store the number of "wall" predictions for each pose
    allPredictions[tuple(pose)] = wallCount

print("Finished predicting.")

# Remove prediction with 0 walls
allPredictions = {k: v for k, v in allPredictions.items() if v != 0}

# Print all predictions in descending order
print("All predictions in descending order:")
for key, value in sorted(allPredictions.items(), key=lambda item: item[1], reverse=True):
    print("%s: %s" % (key, value))