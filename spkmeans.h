#ifndef SPKMEANS_H
#define SPKMEANS_H
#include <math.h>
#include <stdio.h>

/* struct for use in spkmeans.c */
typedef struct eigens{
    int index;
    double value;
    double* vector;
} eigens;

void wam(double** vectors_matrix, int n, int vec_length);
double** wamCalc(double** vectors_matrix, int n, int vec_length);
void ddg(double** vectors_matrix, int n, int vec_length);
double** ddgCalc(double** w_matrix, int n);
void lnorm(double** vectors_matrix, int n, int vec_length);
double** lnormCalc(double** w_matrix, double** d_matrix, int n);
void minusSqrtMatrixD(double** d_matrix, int n);
int eigengapHeuristic(struct eigens* eigens_arr, int n);
void jacobi(double** vectors_matrix, int n);
struct eigens* jacobiCalc(double** a_matrix, int n, int sort);
void copyMatrices(double** mat1, double** mat2, int n);
double** createMatrixI(int n);
void matrixTranspose(double** mat, int n);
int comparator (const void* first, const void* second);
void freeMatrix(double** matrix, int n);
double** createMatrixT(struct eigens* eigens_arr, int n, int k);
double* copyToEigenVectors(double* vec_matrix, int n);
void matrixMult(int rows, int columns, double** mat1, double** mat2, double** result);
double offFunc(double** a_matrix, double** a_tag_matrix, int n);
double** jacobiMatForPrint(struct eigens* eigens_arr, int n);
void printJacobi(struct eigens* eigens_arr, int n);
void printMatrix(double** matrix, int n, int vec_length);
double** allocateMem(int n, int vec_length);
void errorOccured();
double* retrieveLargestIndexesCS(double** a_matrix, int n);
double retrieveTheta(double** a_matrix, int lar_i, int lar_j);
double retrieveT(double theta);
double retrieveC(double t);
double retrieveS(double t, double c);
int sign(double theta);
double** createMatrixP(int n, double* variables);
void updateMatrixAtag(double** a_tag_mat,double** a_mat,int n,double* variables);
void getFinalCentroids(double** centroids,double** elements,int k,int d,int n,int max_iter,double eps);
void resetCentroids(double** centroids,int k, int d);
void initClusters(int* elements_loc, int* items_number_clusters, int n, int k);
void saveCentroids(double** old_centroids, double** centroids, int k, int d);
void assignCentroids(double** ele,double** cntrds,int* in_clstrs,int* ele_loc,int k,int d,int n);
void updateCentroids(double** cntrds,double** ele,int* in_clstrs,int *ele_loc,int d,int n,int k);
int convergence(double** old_centroids,double** centroids,int k,int d,int eps);
#endif 