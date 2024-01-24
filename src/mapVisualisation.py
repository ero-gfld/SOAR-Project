#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

COULOUR_SCHEME = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "lightblue": "#8DA3B3",
    "lightgrey": "#CBC0D5",
    "twgrey": "#72777A"
}

class MapVisualisation:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def showMapWithMainPath(wallPositions, nodesPositions, robotPosition, scan, path):
        plt.rcParams['figure.figsize'] = [7, 7]
        fig, ax = plt.subplots()
        scan = np.array(robotPosition) + scan
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