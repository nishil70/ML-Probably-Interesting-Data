# coding: utf-8
# 
#     EECS 738
#     HW 1: Probably Interesting Data
#     File: KMeansUtility.py
#     Implementation of KMeans Algorithms
# 


from scipy.spatial import distance
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Calculate Euclidean Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def randomCentroids(k,data):
    # Generating ramdom centroids using sample space of the data
    C_x = np.random.randint(np.min(data[:,0]), np.max(data[:,0]), size=k)
    C_y = np.random.randint(np.min(data[:,0]), np.max(data[:,1]), size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    return C

def plotCentroids(C,data):
   
    # Plot the centroids along with data
    plt.scatter(data[:,0], data[:,1], c='#050505', s=7)
    plt.scatter(C[:,0], C[:,1], marker='*', s=200, c='crimson')
    return

def KMeans(k,data,MaxIter):
    centriods = randomCentroids(k,data)
    # Initializing Centroids array
    oldCentroids = np.zeros(centriods.shape)
    # Initializing Cluster array
    clusters = np.zeros(len(data))
    # Error calculted using Euclidean distance function
    error = dist(centriods, oldCentroids, None)
    # Loop will run until maxIter
    for p in range(MaxIter):
    # Assigning each value to its closest cluster
        for i in range(len(data)):
            distances = dist(data[i], centriods)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        oldCentroids = deepcopy(centriods)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            centriods[i] = np.mean(points, axis=0)
        error = dist(centriods, oldCentroids, None)
        return clusters, centriods    

def printClusters(k,data,clusters,centriods):
    colors = ['b', 'g', 'r', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(centriods[:, 0], centriods[:, 1], marker='*', s=200, c='crimson')

