from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset

# Make 3  clusters
k = 3
# Initial Centroids
C = [[2., 0., 3., 4.], [1., 2., 1., 3.], [0., 2., 1., 0.]]
C = np.array(C)
print("Initial Centers")
print(C)
# print(type(C))


def getInput(filename):
    mydata = pd.read_csv(filename, sep=',')
    mydata = mydata.values[:, 0:4].astype('float64')
    return mydata


def k_means(C):
    # Write your code here!
    filename = "data/iris.data"
    mydata = getInput(filename)
    err = 1

    while err >= 0.01:
        cluster1 = []
        cluster2 = []
        cluster3 = []
        distances = np.arange(mydata.shape[0]).reshape(mydata.shape[0], -1)
        for c in C:
            distance_helper = mydata - c
            distance = np.linalg.norm(distance_helper, axis=1).reshape(-1, 1)
            distances = np.column_stack((distances, distance))
        distances = distances[:, 1:]
        minindexs = np.argmin(distances, axis=1)
        for i in range(minindexs.shape[0]):
            if minindexs[i] == 0:
                cluster1.append(mydata[i, :])
            elif minindexs[i] == 1:
                cluster2.append(mydata[i, :])
            else:
                cluster3.append(mydata[i, :])
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
        cluster3 = np.array(cluster3)
        mean1 = np.mean(cluster1, axis=0)
        mean2 = np.mean(cluster2, axis=0)
        mean3 = np.mean(cluster3, axis=0)
        newC = [mean1, mean2, mean3]
        newC = np.array(newC)
        err = np.linalg.norm(C - newC)
        C = newC.copy()
    C_final = C
    return C_final

final_center = k_means(C)
print("Final Centers")
print(final_center)
