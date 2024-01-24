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

ALLOW_VISUALISATION = True
COULOUR_SCHEME = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "lightblue": "#8DA3B3",
    "lightgrey": "#CBC0D5",
    "twgrey": "#72777A"
}

def getMap() -> OccupancyGrid:
    """ Loads map from map service """
    # Create service proxy
    get_map = rospy.ServiceProxy('static_map', GetMap)
    # Call service
    recMap = get_map()
    recMap = recMap.map
    # Return
    return recMap

def showMap(freePositions, wallPositions, scanPositions):
    plt.rcParams['figure.figsize'] = [7, 7]
    fig, ax = plt.subplots()
    ax.scatter(scanPositions[:,1], scanPositions[:,0], c="r", alpha=0.8, label="Laserscan")
    ax.scatter(wallPositions[:,1], wallPositions[:,0], c=COULOUR_SCHEME["darkblue"], alpha=1.0, s=6**2, label="Walls")
    ax.scatter(freePositions[:,1], freePositions[:,0], c=COULOUR_SCHEME["twgrey"], alpha=0.08, s=6**2, label="Unobstructed Space")
    ax.scatter([0], [0], c=COULOUR_SCHEME["twblue"], s=15**2, label="Scan Center")
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Map and Laserscan Data Transformed into World Coordinates")
    ax.set_xticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_yticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    plt.show()

def visualiseClf(clf, X):
    plt.rcParams['figure.figsize'] = [5, 5]
    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=ListedColormap(
            [
                COULOUR_SCHEME["twgrey"],
                COULOUR_SCHEME["darkblue"]
            ]
        ),
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel="X-Coordinate[m]",
        ylabel="Y-Coordinate[m]",
        shading="auto",
    )
    ax.set_title("Fitted kNN Model representing the Map")
    plt.show()

def showMapWithAllPaths(wallPositions, nodesPositions, robotPosition, graph):
    plt.rcParams['figure.figsize'] = [7, 7]
    fig, ax = plt.subplots()
    edgeLines = np.array(
        [
            [
                edge["parent"],
                edge["child"]
            ] for edge in graph
        ]
    )
    ax.scatter(wallPositions[:,1], wallPositions[:,0], c=COULOUR_SCHEME["darkblue"], alpha=1.0, s=6**2, label="Walls")
    ax.scatter(nodesPositions[:,1], nodesPositions[:,0], c=COULOUR_SCHEME["twblue"], alpha=1.0, s=8**2, label="Graph")
    ax.scatter([robotPosition[1]], [robotPosition[0]], c=COULOUR_SCHEME["twblue"], s=15**2, label="Robot Position")
    for line in edgeLines:
        x0, y0 = line[0]
        x1, y1 = line[1]
        x = [x0, x1]
        y = [y0, y1]
        ax.plot(y, x, c=COULOUR_SCHEME["twblue"])
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Graph Generated based on Map Data")
    ax.set_xticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_yticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    plt.show()

def showMapWithMainPath(wallPositions, nodesPositions, robotPosition, scan, path):
    plt.rcParams['figure.figsize'] = [7, 7]
    fig, ax = plt.subplots()
    scan = np.array(bestPose) + scanPositions
    edgeLines = np.array(
        [
            [
                path[index-1],
                path[index]
            ] for index in range(1, len(path))
        ]
    )
    ax.scatter(wallPositions[:,1], wallPositions[:,0], c=COULOUR_SCHEME["darkblue"], alpha=1.0, s=6**2, label="Walls")
    ax.scatter(nodesPositions[:,0], nodesPositions[:,1], c=COULOUR_SCHEME["twblue"], alpha=1.0, s=8**2, label="Path")
    ax.scatter([robotPosition[1]], [robotPosition[0]], c=COULOUR_SCHEME["twblue"], s=15**2, label="Robot Position")
    ax.scatter(scan[:,1], scan[:,0], c="r", alpha=0.8, label="Laserscan")
    for line in edgeLines:
        x0, y0 = line[0]
        x1, y1 = line[1]
        x = [x0, x1]
        y = [y0, y1]
        ax.plot(y, x, c=COULOUR_SCHEME["twblue"])
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Found Path from Robot Position to Exit")
    ax.set_xticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_yticks = [-1, 0, 1, 2, 3, 4 ]
    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    plt.show()

# ----------------------------------------------------------------

# Initiate ROS node
rospy.init_node('localisation')

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

# Align positions to grid (0.25)
wallPositions = np.round(wallPositions / 0.25) * 0.25
freePositions = np.round(freePositions / 0.25) * 0.25

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
if ALLOW_VISUALISATION:
    showMap(freePositions, wallPositions, scanPositions)

# ---------------------------------
# Step 2: Localisation using kNN
# ---------------------------------

# Use KNN to find the nearest neighbour for each point in the scan
knn = neighbors.KNeighborsClassifier(n_neighbors=1)

# X will be all the coordinates of the map's points and Y will be the class of the point (0 for free, 1 for wall)
X = np.concatenate((wallPositions, freePositions))
Y = np.concatenate((np.ones(len(wallPositions)), np.zeros(len(freePositions))))

# Fit the model
clf = knn.fit(X, Y)

# Visualise the model
if ALLOW_VISUALISATION: 
    visualiseClf(clf, X)

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

# Remove prediction with 0 walls
allPredictions = {k: v for k, v in allPredictions.items() if v != 0}

# Print all predictions in descending order
if ALLOW_VISUALISATION:
    print("All predictions in descending order:")
    for key, value in sorted(allPredictions.items(), key=lambda item: item[1], reverse=True):
        print("%s: %s" % (key, value))

# Get the pose with the most walls
bestPose = max(allPredictions, key=allPredictions.get)

# ------------------------------
# Step 3: Exit Maze using Search
# ------------------------------

# Define all the nodes using the "poses" array used earlier
nodes = np.array(poses)

# Define the edges between nodes
edges = np.array([])
for node in nodes:
    # Get all neighbours of the current node
    neighbours = np.array([]) 
    for neighbour in nodes:
        if np.linalg.norm(node - neighbour) == 1:
            neighbours = np.append(neighbours, neighbour)
    # Convert all values to integers and reshape to be 2D
    neighbours = np.reshape(neighbours.astype(int), (-1, 2))
    # Create an edge between the current node and each neighbour if there's no wall in between
    for neighbour in neighbours:
        wallFound = False
        # We check the X and Y axis for walls
        for x in np.arange(node[0], neighbour[0], node[0] > neighbour[0] and -0.25 or 0.25):
            if [x, node[1]] in wallPositions.tolist():
                wallFound = True
                break
        for y in np.arange(node[1], neighbour[1], node[1] > neighbour[1] and -0.25 or 0.25):
            if [node[0], y] in wallPositions.tolist():
                wallFound = True
                break
        # If no wall was found, add the edge
        if not wallFound:
            edges = np.append(edges, {"parent": node, "child": neighbour})

if ALLOW_VISUALISATION:
    showMapWithAllPaths(wallPositions, nodes, bestPose, edges)

# custom dfs recursive implementation
discoveredNodes = []
reversePath = []
currentNode = bestPose
goal = [0, 0]
def MyDFS(currentNode):
    # Add current node to discovered nodes
    print("|_ Entering on node", currentNode)
    discoveredNodes.append(currentNode)
    # We verify if the current node is the goal
    print("  | Verifying if is goal", currentNode, goal, np.array_equal(goal, currentNode))
    if np.array_equal(goal, currentNode): return True
    # We get all children of the current node
    children = [tuple(edge["child"]) for edge in edges if np.array_equal(edge["parent"], currentNode)]
    print("  |_ Getting children", children)
    # We iterate over all children
    for child in children:
        print("    |_ Inspecting child", child)
        # We verify if the child has already been discovered
        if child not in discoveredNodes:
            # We recursively call the function on the child...
            if not MyDFS(child):
                # ...if it returns false, we continue to the next child...
                continue
            # ...else we found the goal and we add the current node to the path
            reversePath.append(child)
            # If the current node is the initial node, we add it to the path
            if currentNode == bestPose:
                reversePath.append(currentNode)
            return True
    return False

# Call the function
MyDFS(currentNode)

# Reverse the path
path = reversePath[::-1]
print("Path found:", path)

# Show the map with the path
if ALLOW_VISUALISATION:
    showMapWithMainPath(wallPositions, nodes, bestPose, scanPositions, path)