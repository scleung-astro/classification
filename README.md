# classification

This repo stores the SimpleKMean class for doing the unsupervised learning by K-mean clustering. 
The class contains the following methods and functions.

1. __init__: To initialize the K-mean solver with given parameters;

__Input__: 

    a. n_clusters: number of clusters to be found, default = 3
    
    b. n_iters: number of maximum iteration to be run, default = 100
    
    c. tolerance: minimum changes of centroid before the solution is accepted
    
__Output__: None

2. fit: To find the centroid based on the cluster numbers, with k-mean++ for initialization

__Input__: 

    a. positions: a (N,2) array containing the positions of N datapoints
    
__Output__: None

3. predict: To find the nearest cluster for given points

__Input__:

    a. position: a (N,2) array containing the position of datapoint to be classified
    
__Output__: 

    a. An (N,1) array of which cluster the given points belong

4. plot: To plot the classification

__Input__: None

__Output__: 

    a. The pyplot GUI of the figure plotted based on the fitting data in fit

