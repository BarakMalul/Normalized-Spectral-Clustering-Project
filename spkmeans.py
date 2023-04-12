import sys
import numpy as np
import spkmeansmodule as spkmm
np.random.seed(0) # set the seed at the beginning of the code

# K-means++ algorithm
def initializeCentroids(vectors, k, n):
    centroids = []
    random_index = np.random.randint(n)
    centroids.append(vectors[random_index])
    indexes_used = [random_index]
    index_list=[x for x in range(n)]
    i = 1
    while (i<k):
        distances = retrieveDistances(vectors, centroids)
        prob = retrieveProb(distances)
        new_index = (np.random.choice(index_list, 1, p=prob))[0]
        centroids.append(vectors[new_index])
        indexes_used.append(new_index)
        i += 1
    return centroids, indexes_used

# distances for init centroids
def retrieveDistances(vectors, centroids):
    n = len(vectors)
    distances = [0 for i in range(n)]
    for j in range(n):
        min=float('inf')
        for q in range(len(centroids)):
            dist = distanceCalc(vectors[j], centroids[q])
            if dist < min:
                min = dist
        distances[j] = min
    return distances

def distanceCalc(x, y):
    dist=0
    for i in range(len(x)):
        dist+=(float(x[i])-float(y[i]))**2
    return dist

# probabilities for init centroids
def retrieveProb(distances):
    n = len(distances)
    Sum = sum(distances)
    prob = [0.0 for i in range(n)]
    for i in range(n):
        prob[i] = distances[i]/Sum
    return prob

# flatten mat is a mat in an array form, for C-API
def retrieveFlattenMat(mat,rows,columns):
    lst = []
    for i in range(rows):
        for j in range(columns):
            lst.append(float(mat[i][j]))
    return lst

# print matrices in python
def printMatrix(mat, rows, columns):
    for i in range(rows):
        for j in range(columns):
            mat[i][j] = '%.4f'%mat[i][j]
        result = []
    for i in range(rows):
        result.append(",".join(mat[i]))
    for i in range(rows):
            print(result[i])

def printSPK(final_centroids,indexes, k):    
    for i in range(len(indexes)):
        indexes[i] = str(indexes[i])
    indexes = ",".join(indexes)
    print(indexes)
    printMatrix(final_centroids, k, k)

# main 
if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("Invalid Input!")
        sys.exit()
    # read arguments
    try:
        k = int(sys.argv[1]) 
        max_iter = 300
        goal = sys.argv[2]
        possible_goals = {"wam","ddg","lnorm","jacobi","spk"}
        if goal not in possible_goals:
            print("Invalid Input!")
            sys.exit()
        # read the input file
        input = np.loadtxt(sys.argv[3], delimiter=',')
    except:
        print("Invalid Input!")
        sys.exit()

    n = input.shape[0]
    d = input.shape[1]
    # check for validity of k input
    if ((k==1 or k<0) and goal=="spk") or k>=n: 
        print("Invalid Input!")
        sys.exit()
    
    # send the input to get the final matrix from C, by the desired goal
    flatten_input = input.flatten().tolist()
    final_mat = spkmm.getMatrixByGoal(k, n, d, flatten_input, goal)
    rows = len(final_mat)
    columns = len(final_mat[0])

    # if goal is not spk, print the matrix from C as is
    if goal != "spk":
        printMatrix(final_mat, rows, columns)

    # spk: T matrix is recieved by C, sent to fit function, Kmeans in C
    else:
        k = columns
        centroids, indexes = initializeCentroids(final_mat, k, n)
        
        # preparing the matrices for input to C
        initial_centroids = retrieveFlattenMat(centroids, k, k)
        t_matrix = retrieveFlattenMat(final_mat, n, k)
        
        # final centroids received by fit functions 
        final_centroids = spkmm.fit(k, n, k, max_iter, initial_centroids, t_matrix)
        
        # prints the initial indexes and the final centroids
        printSPK(final_centroids, indexes, k)