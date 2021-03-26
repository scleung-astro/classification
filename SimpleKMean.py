'''
This class contains the Simple K-Mean algorithm for unsupervised learning.
It contains the following methods:

1. __init__: To initialize the K-mean solver with given parameters;
Input: 
    a. n_clusters: number of clusters to be found, default = 3
    b. n_iters: number of maximum iteration to be run, default = 100
    c. tolerance: minimum changes of centroid before the solution is accepted
Output: None

2. fit: To find the centroid based on the cluster numbers, with k-mean++ for initialization
Input: 
    a. positions: a (N,2) array containing the positions of N datapoints
Output: None

3. predict: To find the nearest cluster for given points
Input:
    a. position: a (N,2) array containing the position of datapoint to be classified
Output: 
    a. An (N,1) array of which cluster the given points belong

4. plot: To plot the classification
Input: None
Output: 
    a. The pyplot GUI of the figure plotted based on the fitting data in fit

Written by Shing Chi Leung
'''

from math import sqrt
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class SimpleKMean():

    def __init__(self, n_clusters=3, n_iters=20, tolerance=0.01):

        # feed in the input parameter
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.tolerance = tolerance

        # centroid to be found
        self.centroids = [None for i in range(n_clusters)]
        self.classification = None
        self.raw_data = None

    def fit(self, X):

        # Initialization
        # use K-mean++ to make the initial guess. This means
        # 1. Select a random point as the first centroid
        # 2. Find the distance D of all points from their nearest centroid
        # 3. Choose a random new centroid with a probability D^2
        # 4. Repeat 2 and 3 until all initial centroids are determined

        # copy the data to the class for later use
        self.raw_data = [(x[0], x[1]) for x in X]

        # number of datapoints in the input array
        num_points = len(X)

        # step 1: The first centroid is pure guess
        idx = rd.randint(0, num_points-1)
        self.centroids[0] = [X[idx][0], X[idx][1]]

        # Other centroid is random guess with a prob distribution
        for i in range(1, self.n_clusters):

            dmin_sq = [np.inf for i in range(num_points)]
            for j, point in enumerate(X):

                # step 2
                # for each point, loop over determined centroid to find the minimum distance squared
                # only determined centroids are looped
                for k in range(i):
                    dist2_new = (self.centroids[k][0] - point[0])**2 + (self.centroids[k][1] - point[1])**2
                    if  dist2_new < dmin_sq[j]:
                        dmin_sq[j] = dist2_new

            dmin_sq_sum = np.sum(dmin_sq)
            # normalize the probability distribution
            dmin_sq = [k / dmin_sq_sum for k in dmin_sq]

            # step 3 use the prob distribution to find the next centroid
            idx = np.random.choice(num_points, size=1, p=dmin_sq)
            self.centroids[i] = [X[idx[0]][0], X[idx[0]][1]]

        # now do the standard k-mean
        # cluster idx is the current cluster the point belongs to 
        cluster_idx = [None for i in range(num_points)]


        # steps for Simple K-mean
        # 1. Classify all points to find their nearest cluster (cluster_idx)
        # 2. Find the new centroid based on the results
        # 3. Check if the movement of centroid is smaller then threshold
        # 3b. If yes, then leave the loop, otherwise repeat from Step 1
        
        for n in range(self.n_iters):

            # dummy variables to be used in the loop
            new_centroids = [[0,0] for i in range(self.n_clusters)]

            # total number of points belonging to that cluster
            cluster_counts = [0 for i in range(self.n_clusters)]

            # classify the points according to their distance to the nearest centroids
            for i, point in enumerate(X): 
                dmin_sq = np.inf
                for j, centroid in enumerate(self.centroids):
                    if (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2 < dmin_sq:
                        cluster_idx[i] = j
                        dmin_sq = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2       
                             
            # Step 2: Find the new centroid
            for i, point in enumerate(X):
                new_centroids[cluster_idx[i]][0] += point[0]
                new_centroids[cluster_idx[i]][1] += point[1]
                cluster_counts[cluster_idx[i]] += 1

            # take average by definition of centroid
            for i in range(self.n_clusters):
                new_centroids[i][0] /= cluster_counts[i]
                new_centroids[i][1] /= cluster_counts[i]

            #print(n, new_centroids, self.centroids)

            # Step 3: Check the movement of centroid if they exceed the threshold
            below_threshold = True
            for i in range(self.n_clusters):
                if sqrt((new_centroids[i][0] - self.centroids[i][0])**2 + (new_centroids[i][1] - self.centroids[i][1])**2) > \
                    self.tolerance:
                    below_threshold = False
                    
            # update the new position of the centroids
            for i in range(self.n_clusters):
                self.centroids[i][0] = new_centroids[i][0]
                self.centroids[i][1] = new_centroids[i][1]

            if below_threshold:
                break

        # copy the classification from the fitting for later use
        self.classification = [idx for idx in cluster_idx]

        # do a quick output to test the result
        print("The centroids are located at:")
        for centroid in self.centroids:
            print("x = {}, y = {}".format(centroid[0], centroid[1]))


    def predict(self, points):

        cluster_idx = [None for i in range(len(points))]

        for i, point in enumerate(points): 
            dmin_sq = np.inf
            for j, centroid in enumerate(self.centroids):
                d2 = (centroid[0] - point[0])**2 + (centroid[1] - point[1])**2
                if d2 < dmin_sq:
                    dmin_sq = d2
                    cluster_idx[i] = j

        return cluster_idx


    def plot(self):

        # extract the data 
        datapoints = np.array(self.raw_data)
        centroids = np.array(self.centroids) 

        plt.rcParams.update({"font.size":14})

        fig, ax = plt.subplots(figsize=(8,8), ncols=1, nrows=1)
        ax.scatter(datapoints[:,0], datapoints[:,1], c=self.classification, s=100)
        ax.scatter(centroids[:,0], centroids[:,1], c=range(self.n_clusters), s=100, marker="x")
        
        ax.set_title("Classified Data (Circle: Datapoint, cross: centroid)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()

# here I demonstrate how to use the SimpleKMean class                
def main():

    # generate random points
    nop = 100

    X = [(rd.random(), rd.random()) for i in range(nop)]

    # initialize and fit the data
    skmean = SimpleKMean(n_clusters=3, n_iters=30, tolerance=0.01)
    skmean.fit(X)

    # to classify a new data point by the given centroids
    test_data = [(0.5,0.5), (0.75,0.75)]
    results = skmean.predict(test_data)

    print("Test the prediction")
    for i, data in enumerate(test_data):
        print("x = {}, y = {}, cluster = {}".format(data[0], data[1], results[i]))

    # to plot the fitted data and the corresponding centroid
    skmean.plot()


if __name__=="__main__":
    main()

