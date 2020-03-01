import sys
import scipy.io.wavfile
import numpy as np


# Calculate distance between two points
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Checks which centroid is closest to the point and returning its index
def find_closest_centroid_index(x_arg, y_arg, centroids_arg):
    minDis = distance(x_arg, y_arg, centroids_arg[0][0], centroids_arg[0][1])
    closestCentroidIndex = 0
    for i in range(len(centroids_arg)):
        dis = distance(x_arg, y_arg, centroids_arg[i][0], centroids_arg[i][1])
        if dis < minDis:
            minDis = dis
            closestCentroidIndex = i
    return closestCentroidIndex


# Updates the centroids to be the average of the points that belong to them
def update_centroids(x_arg, y_arg, centroids_arg, classifications_arg):
    for i in range(len(centroids_arg)):
        xSum = 0
        ySum = 0
        numOfPoints = 0
        for j in range(len(classifications_arg)):
            if classifications_arg[j] == i:
                xSum = xSum + x_arg[j]
                ySum = ySum + y_arg[j]
                numOfPoints = numOfPoints + 1
        if numOfPoints == 0:
            break
        centroids_arg[i][0] = xSum / numOfPoints
        centroids_arg[i][1] = ySum / numOfPoints


# Implementation of the k-means algorithm
def k_means(x_arg, y_arg, centroids_arg, classifications_arg, new_values_arg):
    f = open("output.txt", "w")

    counter = 0
    for i in range(len(x_arg)):
        classifications_arg[i] = find_closest_centroid_index(x_arg[i], y_arg[i], centroids_arg)
    old_centroids = np.array(centroids_arg.copy())
    update_centroids(x_arg, y_arg, centroids_arg, classifications_arg)
    centroids_arg = centroids_arg.round()
    f.write(f"[iter {counter}]:{','.join([str(i) for i in centroids_arg])}\n")
    counter = counter + 1

    while not np.array_equal(old_centroids, centroids_arg):
        if counter == 30:
            break
        for j in range(len(x_arg)):
            classifications_arg[j] = find_closest_centroid_index(x_arg[j], y_arg[j], centroids_arg)
        old_centroids = np.array(centroids_arg.copy())
        update_centroids(x_arg, y_arg, centroids_arg, classifications_arg)
        centroids_arg = centroids_arg.round()
        f.write(f"[iter {counter}]:{','.join([str(i) for i in centroids_arg])}\n")
        counter = counter + 1

    f.close()

    # Updates each point to be the value of the nearest centroid
    for k in range(len(new_values_arg)):
        new_values_arg[k][0] = centroids_arg[classifications_arg[k]][0]
        new_values_arg[k][1] = centroids_arg[classifications_arg[k]][1]


sample, centroids = sys.argv[1], sys.argv[2]
fs, arr = scipy.io.wavfile.read(sample)
new_values = np.array(arr.copy())
x, y = zip(*arr)
centroids = np.loadtxt(centroids)
classifications = [0] * len(x)

if len(centroids) == 0:
    exit(-1)

k_means(x, y, centroids, classifications, new_values)

scipy.io.wavfile.write("compressed.wav", fs, np.array(new_values, dtype=np.int16))
