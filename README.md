# Normalized Spectral Clustering Algorithm

2022, Software Project Course

Tel Aviv University

•	An algorithm which groups datapoints that are retrieved from matrices into clusters by using centroids (mean value of the datapoints in the cluster) for each cluster.

•	The main algorithm uses mathematical algorithms: creation of adjacency matrix, diagonalization to find eigenvectors, arranging the eigenvectors in a matrix as columns and renormalizing. A clustering algorithm (K-means) runs on the last matrix. The algorithm adjusts the centroids each iteration and repeats itself until convergence.

•	Using Python C-API to implement C code for the main computational work of the algorithm in the Python code. working in a pair, coordinated using GitHub.
