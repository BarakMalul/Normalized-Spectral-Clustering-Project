#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "spkmeans.h"
#define EPSILON 0.00001

int main(int argc, char* argv[]){
    int n, i, j, r;
    int vec_length;
    char* goal;
    char* input_filename;
    double** vectors_matrix;
    FILE* input_file;
    double vector_item;
    char tav;

    if (argc != 3){
        printf ("Invalid Input!");
        exit(1);
    }
    goal = argv[1]; /* the enum type */
    input_filename = argv[2];

    input_file = fopen(input_filename, "r");
    if (input_file == NULL)
        errorOccured();

    /* counts the dimension(vec_length), and the number of vectors(n) */
    vec_length = 1;
    n = 0;
    tav = fgetc(input_file);
    while (tav != '\n'){
        if (tav == ',')
            vec_length++;
        tav = fgetc(input_file);
    }
    while (tav != EOF){
        if (tav == '\n')
            n++;
        tav = fgetc(input_file);
    }
    fseek(input_file,0,0); /* rewind file */

    vectors_matrix = allocateMem(n, vec_length);
    /* insert vectors to vectors_matrix */
    for (i = 0; i < n; i++){
        for (j = 0; j < vec_length; j++){
            fscanf(input_file, "%lf%c", &vector_item, &tav);
            vectors_matrix[i][j] = vector_item;
        }
    }
    r = fclose(input_file);
    if (r != 0){
        errorOccured();
    }
    /* check which goal to choose */
        if (strcmp(goal, "wam") == 0){
            wam(vectors_matrix, n, vec_length);
        }
        else if (strcmp(goal, "ddg") == 0){
            ddg(vectors_matrix, n, vec_length);
        }
        else if (strcmp(goal, "lnorm") == 0){
            lnorm(vectors_matrix, n, vec_length);
        }
        else if (strcmp(goal, "jacobi") == 0){
            jacobi(vectors_matrix, n);
        }
        else {
            printf ("Invalid Input!");
            exit(1);
        }
        
    freeMatrix(vectors_matrix, n);

    return 0;
}

void wam(double** vectors_matrix, int n, int vec_length){
    double** w_matrix;
    w_matrix = wamCalc(vectors_matrix, n, vec_length);
    printMatrix(w_matrix, n, n);
    freeMatrix(w_matrix, n);
}

double** wamCalc(double** vectors_matrix, int n, int vec_length){
    int i;
    int j;
    int s;
    double sum;
    double** w_matrix;
    w_matrix = allocateMem(n, n);

    for (i = 0; i < n; i++){
        for (j= i + 1; j < n; j++){
            sum = 0;
            for (s = 0; s < vec_length; s++)
                sum += pow((vectors_matrix[i][s] - vectors_matrix[j][s]),2);
            w_matrix[i][j] = exp((-1)*(sqrt(sum)/2));
            w_matrix[j][i] = exp((-1)*(sqrt(sum)/2)); /* w_matrix is symetric */
        }
    }
    return w_matrix;
}

void ddg(double** vectors_matrix, int n, int vec_length){
    double** w_matrix;
    double** d_matrix;

    w_matrix = wamCalc(vectors_matrix, n, vec_length);
    d_matrix = ddgCalc(w_matrix, n);
    printMatrix(d_matrix, n, n);

    freeMatrix(w_matrix, n);
    freeMatrix(d_matrix, n);
}

double** ddgCalc(double** w_matrix, int n){
    double** d_matrix;
    int i,j;
    d_matrix = allocateMem(n, n);

    /* the sum of each row goes in the diagonal line of d_matrix */
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++)
            d_matrix[i][i] += w_matrix[i][j];
    }

    return d_matrix;
}
        
void lnorm(double** vectors_matrix, int n, int vec_length){
    double** d_matrix;
    double** w_matrix;
    double** lnorm_matrix;

    w_matrix = wamCalc(vectors_matrix, n, vec_length);
    d_matrix = ddgCalc(w_matrix, n);
    lnorm_matrix = lnormCalc(w_matrix, d_matrix, n);
    printMatrix(lnorm_matrix, n, n);

    freeMatrix(w_matrix, n);
    freeMatrix(d_matrix, n);
    freeMatrix(lnorm_matrix, n);
}

double** lnormCalc(double** w_matrix, double** d_matrix, int n){
    int i;
    int j;
    double** laplacian_matrix;
    double** i_matrix;
    double** result_matrix;
    double** temp_matrix;
    laplacian_matrix = allocateMem(n, n);
    result_matrix = allocateMem(n, n);
    temp_matrix = allocateMem(n, n);
    i_matrix = createMatrixI(n);

    /* calculate D^(-0.5) */
    minusSqrtMatrixD(d_matrix, n);
    
    /* calculation of D^(-1/2) * W * D^(-1/2): */
    /* temp matrix = D^(-1/2) * W */
    matrixMult(n,n,d_matrix,w_matrix,temp_matrix);
    /* result matrix = (D^(-1/2) * W) * D^(-1/2) */
    matrixMult(n,n,temp_matrix,d_matrix,result_matrix);

    /*calculate final L_norm */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            laplacian_matrix[i][j] = i_matrix[i][j] - result_matrix[i][j];

    freeMatrix(i_matrix, n);
    freeMatrix(result_matrix, n);
    freeMatrix(temp_matrix, n);

    return laplacian_matrix;
}

/* D = D^(-0.5) */
void minusSqrtMatrixD(double** d_matrix, int n){
    int i;

    for (i = 0; i < n; i++)
        d_matrix[i][i] = 1/sqrt(d_matrix[i][i]);
}

/* use of 1.3 The Eigengap Heuristic when input k is 0 */
int eigengapHeuristic(struct eigens* eigens_arr, int n){
    double* deltas;
    int i, argmax_i;
    double max_delta;
    deltas = (double*)calloc(n-1, sizeof(double));
    if (deltas == NULL)
        errorOccured();

    for (i = 0; i < n - 1; i++)
        deltas[i] = fabs(eigens_arr[i].value - eigens_arr[i+1].value);
        
    /* k = argmax_i(delta_i), i = 1,...,n/2 */
    max_delta = deltas[0];
    argmax_i = 0;
    for (i = 0; i < (int)(n/2); i++){
        if (deltas[i] > max_delta){
            max_delta = deltas[i];
            argmax_i = i;
        }
    }

    free(deltas);
    return (argmax_i+1);
}

void jacobi(double** vectors_matrix, int n){
    int i;
    struct eigens* eigens_arr;

    eigens_arr = jacobiCalc(vectors_matrix, n, 0);
    printJacobi(eigens_arr, n);

    for (i=0; i<n; i++)
        free(eigens_arr[i].vector);
    free(eigens_arr);
}

/* sort == 1 means to do the operation */
struct eigens* jacobiCalc(double** a_matrix, int n, int sort){
    double off;
    int i, rotations_number;
    double** p_matrix;
    double** a_tag_matrix;
    double** v_matrix;
    double** temp_matrix;
    struct eigens* eigens_arr;
    double* variables;
    rotations_number = 0;
    off = 1; /* initialize off, guaranteed first iteration */
    temp_matrix = allocateMem(n,n);
    v_matrix = createMatrixI(n);
    eigens_arr = (eigens*) calloc(n, sizeof(struct eigens));
    if (eigens_arr == NULL)
        errorOccured();
    a_tag_matrix = allocateMem(n,n);
    copyMatrices(a_tag_matrix, a_matrix, n); /* initialize a_tag */ 

    while((rotations_number < 100) && (off >= EPSILON)){

        /* find latgest item and its i,j, and also c & s needed for calcs */
        variables = retrieveLargestIndexesCS(a_matrix, n);
        rotations_number++;
        
        p_matrix = createMatrixP(n, variables);  

        /* calculate A' according to step 6 */
        updateMatrixAtag(a_tag_matrix, a_matrix, n, variables);
        
        /* temp_matrix = V * P_i */
        matrixMult(n, n, v_matrix, p_matrix, temp_matrix);

        /* V =  temp_matrix */
        copyMatrices(v_matrix, temp_matrix, n);

        /* calc off(A)^2 - off(A')^2 */
        off = offFunc(a_matrix, a_tag_matrix, n); 

        /* A = A' */
        copyMatrices(a_matrix, a_tag_matrix, n);

        freeMatrix(p_matrix, n);
        free(variables);
    } 

    /* transpose for eigenvectors in rows, easier to work with */
    matrixTranspose(v_matrix, n); 

    for (i = 0; i < n; i++){
        eigens_arr[i].index = i;
        eigens_arr[i].value = a_matrix[i][i];
        eigens_arr[i].vector = copyToEigenVectors(v_matrix[i], n);
    }

    freeMatrix(a_tag_matrix, n);
    freeMatrix(temp_matrix, n);
    freeMatrix(v_matrix, n);
    
    if (sort == 1) /* for spk goal, sort in a decreasing order */
        qsort(eigens_arr, n, sizeof(struct eigens), comparator);

    return eigens_arr;
}

/* copies from mat2 to mat1 */
void copyMatrices(double** mat1, double** mat2, int n){
    int i,j;

    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            mat1[i][j] = mat2[i][j];
        }
    }
}

/* creates the identity matrix */
double** createMatrixI(int n){
    double** i_matrix;
    int i;
    i_matrix = allocateMem(n,n);

    for (i=0; i<n; i++)
        i_matrix[i][i] = 1;

    return i_matrix;
}

/* mat = mat^T (transposed) */
void matrixTranspose(double** mat, int n){
    int i, j;
    double** trans_matrix;
    trans_matrix = allocateMem(n, n);

    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            trans_matrix[i][j] = mat[i][j];
        }
    }

    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            mat[j][i] = trans_matrix[i][j];
        }
    }
    
    freeMatrix(trans_matrix, n);
}

/* comparator func to sort eigenvalues in a decreasing order */
int comparator(const void* first, const void* second){
    struct eigens* e1;
    struct eigens* e2; 

    e1 =  (struct eigens*) first;
    e2 =  (struct eigens*) second;

    if (e1->value > e2->value)
        return -1;
    else if (e1->value < e2->value) 
        return 1;
    else if (e1->index < e2->index) 
        return -1;
    return 0;
}

/* frees memory for 2D arrays */
void freeMatrix(double** matrix, int n){
    int i;

    for (i=0; i < n; i++)
        free(matrix[i]);
    free(matrix);
}
/* for spk in cmodule, creates U and renormalizing */
double** createMatrixT(struct eigens* eigens_arr, int n, int k){
    int i;
    int j;
    double sum;
    double** t_matrix;
    double** u_matrix;
    t_matrix = allocateMem(n, k);
    u_matrix = allocateMem(n, k);

    for (i = 0; i < k; i++){
        for (j = 0; j < n; j++){
             u_matrix[j][i] = eigens_arr[i].vector[j];
        }
    }

    for (i = 0; i < n; i++){
        sum = 0;
        for (j = 0; j < k; j++)
            sum += pow(u_matrix[i][j], 2);
        sum = sqrt(sum);

        for (j = 0; j < k; j++){
            if (sum != 0)
                t_matrix[i][j] = ((u_matrix[i][j]) / (sum));
            else
                t_matrix[i][j] = 0.0;
        }
    }

    freeMatrix(u_matrix, n);

    return t_matrix;
}

/* makes a vector out of a row in a matrix */
double* copyToEigenVectors(double* vec_matrix, int n){
    int i;
    double* vector;
    vector = (double*) calloc(n, sizeof(double));
    if (vector == NULL)
        errorOccured();
    
    for (i = 0; i < n; i++)
        vector[i] = vec_matrix[i];
    
    return vector;
}

/* calculates the multiplication of two matrices into result (matrix) */
void matrixMult(int rows,int columns,double** mat1,double** mat2,double** result){
    int i, j, k;

    for (i = 0; i < rows; i++){
        for (j = 0; j < columns; j++){
            result[i][j] = 0;
            for (k = 0; k < rows; k++)
                result[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
}

/* calculates off(A)^2 - off(A')^2 */
double offFunc(double** a_matrix, double** a_tag_matrix, int n){
    double off_a, off_a_tag;
    int i, j;
    off_a = 0;
    off_a_tag = 0;

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (i==j)
                continue;
            off_a += pow(a_matrix[i][j], 2); 
            off_a_tag += pow(a_tag_matrix[i][j], 2);
        }
    }

    return (off_a - off_a_tag);
}

double** jacobiMatForPrint(struct eigens* eigens_arr, int n){
    double** jacobi_matrix;
    int i, j;
    jacobi_matrix = allocateMem(n, n);

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++)
            jacobi_matrix[j][i] = eigens_arr[i].vector[j];
    }
    return jacobi_matrix;
}

void printJacobi(struct eigens* eigens_arr, int n){
    int i;
    double** jacobi_matrix;

    for (i = 0; i < n; i++){
         if (eigens_arr[i].value < 0 && eigens_arr[i].value > -0.00005)
            printf("0.0000");
        else
            printf("%.4f", eigens_arr[i].value);
        if (i != n - 1)
            printf(",");
    }
    printf("\n");
    jacobi_matrix = jacobiMatForPrint(eigens_arr, n);
    printMatrix(jacobi_matrix, n, n);

    freeMatrix(jacobi_matrix, n);
}

void printMatrix(double** matrix, int n, int vec_length){
    int i, j;
    
    for (i = 0; i < n; i++){
        for (j = 0; j < vec_length; j++) {
            if (matrix[i][j] < 0 && matrix[i][j] > -0.00005)
                printf("0.0000");
            else
                printf("%.4f", matrix[i][j]);
            if (j != vec_length - 1)
                printf(",");
        }
        printf("\n");
    }
}

/* allocates memory for a matrix */
double** allocateMem(int n, int vec_length){ 
    int i;
    double** matrix;
    matrix = (double **)calloc(n, sizeof(double *));
    if (matrix == NULL)
        errorOccured();

    for (i = 0; i < n; i++){
        matrix[i] = (double *)calloc(vec_length, sizeof(double));
        if (matrix[i] == NULL){
            freeMatrix(matrix, i); 
            errorOccured();
        }
    }

    return matrix;
}

void errorOccured(){
    printf("An Error Has Occurred");
    exit(1);
}

/* retrieves the indexes of the largest element in the matrix
** and calculates theta, t, c and s, returns largest i&j, c&s */
double* retrieveLargestIndexesCS(double** a_matrix, int n){
    double largest;
    int i, j, lar_i, lar_j;
    double* ret_arr;
    double theta, t;
    lar_i = 0;
    lar_j= 1;
    ret_arr = (double*)calloc(4, sizeof(double));
    if (ret_arr == NULL)
        errorOccured();   
    
    largest = fabs(a_matrix[0][1]);
    for (i = 0; i < n; i++){
        for (j = i + 1; j < n; j++){
            if (fabs(a_matrix[i][j]) > fabs(largest)){
                /* 3. find the Pivot A_ij */
                largest = fabs(a_matrix[i][j]);
                lar_i = i;
                lar_j = j;
            }
        }
    }
    ret_arr[0] = (double)lar_i;
    ret_arr[1] = (double)lar_j;

    if (a_matrix[lar_i][lar_j] == 0){
        ret_arr[2] = 1; /* ret_arr[2] = c */
        ret_arr[3] = 0; /* ret_arr[3] = s */
    }
    else {
        theta = retrieveTheta(a_matrix, lar_i, lar_j);
        t = retrieveT(theta);
        ret_arr[2] = retrieveC(t); /* ret_arr[2] = c */
        ret_arr[3] = retrieveS(ret_arr[2], t); /* ret_arr[3] = s */
    }

    return ret_arr;
}

/* helper functions for retrieveLargestIndexesCS */
double retrieveTheta(double** a_matrix, int lar_i, int lar_j){
    return (a_matrix[lar_j][lar_j] - a_matrix[lar_i][lar_i])/(2*a_matrix[lar_i][lar_j]);
}

double retrieveT(double theta){
    return sign(theta)/(fabs(theta) + sqrt((pow(theta, 2)) + 1));
}

double retrieveC(double t){
    return 1/sqrt((pow(t, 2)) + 1);
}

double retrieveS(double t, double c){
    return t*c;
}

int sign(double theta){
    return theta>=0? 1:-1;
}

double** createMatrixP(int n, double* variables){
    double** p_matrix;
    double c, s;
    int lar_i, lar_j;
    lar_i = (int)variables[0];
    lar_j = (int)variables[1];
    c = variables[2];
    s = variables[3];
    p_matrix = createMatrixI(n);

    p_matrix[lar_i][lar_i] = c;
    p_matrix[lar_j][lar_j] = c;
    p_matrix[lar_i][lar_j] = s;
    p_matrix[lar_j][lar_i] = -s;

    return p_matrix;
}

/* calculates A' according to step 6 */
void updateMatrixAtag(double** a_tag_mat,double** a_mat,int n,double* variables){
    int i, lar_i, lar_j;
    double c, s;
    lar_i = (int)variables[0];
    lar_j = (int)variables[1];
    c = variables[2];
    s = variables[3];

    for (i=0; i<n; i++){
        if ((i != lar_i) || (i != lar_j)){
                a_tag_mat[i][lar_i] = c*a_mat[i][lar_i] - s*a_mat[i][lar_j];
                a_tag_mat[lar_i][i] = c*a_mat[i][lar_i] - s*a_mat[i][lar_j];
                a_tag_mat[i][lar_j] = c*a_mat[i][lar_j] + s*a_mat[i][lar_i];
                a_tag_mat[lar_j][i] = c*a_mat[i][lar_j] + s*a_mat[i][lar_i];
        }
    }
    a_tag_mat[lar_i][lar_i] = c*c*a_mat[lar_i][lar_i] + s*s*a_mat[lar_j][lar_j] - 2*s*c*a_mat[lar_i][lar_j];
    a_tag_mat[lar_j][lar_j] = s*s*a_mat[lar_i][lar_i] + c*c*a_mat[lar_j][lar_j] + 2*s*c*a_mat[lar_i][lar_j];
    a_tag_mat[lar_i][lar_j] = 0;
    a_tag_mat[lar_j][lar_i] = 0;
}

/* K-means Functions */
void getFinalCentroids(double** centroids,double** elements,int k,int d,int n,int max_iter,double eps){
    int converge_bit;
    int iteration_number;
    int* elements_location;
    int* items_number_clusters;
    double** old_centroids;
    converge_bit = 1; 
    iteration_number = 0;
    elements_location = (int*)calloc(n, sizeof(int));
    if (!elements_location)
        errorOccured();
    items_number_clusters = (int*)calloc(k, sizeof(int));
    if (!items_number_clusters)
        errorOccured();
    old_centroids = allocateMem(k, d);

    /* converge_bit == 0 means convergence -> exit while loop */
    while (converge_bit==1 && max_iter>iteration_number){
        iteration_number++;
        initClusters(elements_location, items_number_clusters, n, k);
        assignCentroids(elements, centroids, items_number_clusters,elements_location,k,d,n);
        saveCentroids(old_centroids, centroids, k, d);
        resetCentroids(centroids, k, d);
        updateCentroids(centroids,elements,items_number_clusters,elements_location,d,n,k);
        converge_bit = convergence(old_centroids, centroids, k, d, eps);
    }

    free(elements_location);
    free(items_number_clusters);
    freeMatrix(old_centroids, k);
}

void resetCentroids(double** centroids,int k, int d){
    int i,j;

    for (i=0 ;i<k; i++)
        for (j=0; j<d; j++)
            centroids[i][j]=0.0;                
}

void initClusters(int* elements_loc, int* items_number_clusters, int n, int k){
    int i;

    for (i=0; i<n; i++)
        elements_loc[i]=0;
    for (i=0; i<k; i++)
        items_number_clusters[i]=0;
}

void saveCentroids(double** old_centroids, double** centroids, int k, int d){
    int i,j;

    for (i=0; i<k; i++){
            for (j=0; j<d; j++)
                old_centroids[i][j] = centroids[i][j];
        }
}

void assignCentroids(double** ele,double** cntrds,int* in_clstrs,int* ele_loc,int k,int d,int n){
    int i, j, l;
    double sum=0.0;
    double min;
    int flag;
    int min_index;

    for (i=0; i < n; i++){
        min_index = 0;
        flag = 0;
        for (j=0; j < k; j++){
            sum = 0.0;
                for (l = 0; l < d; l++)
                    sum += pow((ele[i][l] - cntrds[j][l]),2);
            sum = sqrt(sum);
            if (flag == 0){
                min = sum;
                flag = 1;
            }
            else if (sum < min){
                min = sum;
                min_index = j;
            }
        }
        in_clstrs[min_index]+=1;
        ele_loc[i]=min_index;
    }
}

void updateCentroids(double** cntrds,double** ele,int* in_clstrs,int *ele_loc,int d,int n,int k){
    int m, i, q;

    for (i=0; i<k; i++){
            for (m=0; m<n; m++){
                if (ele_loc[m]==i){
                    for (q=0; q<d; q++)
                        cntrds[i][q]+=ele[m][q];
                }
            }
            for (q=0; q<d; q++){
                if (cntrds[i][q]==0){
                    continue;
                }
                cntrds[i][q]=cntrds[i][q]/(double)in_clstrs[i];
            }
        }
}

int convergence(double** old_centroids,double** centroids,int k,int d,int eps){
    int converge_bit;
    int i, j;
    double sum;

    converge_bit = 0;
    for (i=0; i<k; i++){
        sum = 0.0;
        for (j = 0; j<d; j++)
            sum += pow((old_centroids[i][j] + centroids[i][j]), 2);
        sum = sqrt(sum);
        if (sum >= eps)
            converge_bit = 1;
    }
    return converge_bit;
}