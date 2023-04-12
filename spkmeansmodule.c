#define PY_SSIZE_T_CLEAN
#include "spkmeans.h"
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* helper function - saves the output to Python into a list */
static void makeListFromMatrix(PyObject* lst,double** matrix,int n,int k){
    int i, j;
    PyObject* row;

    for (i = 0; i < n; i++){
        row = PyList_New(0);
        if (!row){
            Py_DECREF(lst);
            errorOccured();
        }
        for (j = 0; j < k; j++){
            PyList_Append(row,PyFloat_FromDouble((double)matrix[i][j]));
        }
        PyList_Append(lst, row);
    }  
}

/* helper function - saves the matix input from Python into a C matrix */
static double** makeMatrixFromPyInput(PyObject* py_matrix,int n,int vec_length)
{
    int i,j;
    double** new_matrix;
    PyObject* element;
    new_matrix = allocateMem(n, vec_length);

    for (i = 0; i < n; i++){
        for (j = 0; j < vec_length ; j++) {
            element = PyList_GetItem(py_matrix, (vec_length*i) + j);
            new_matrix[i][j] = PyFloat_AsDouble(element);
        }
    }
    Py_DECREF(py_matrix);
    
    return new_matrix;
}

/**
 * function called from python.
 * performs the spkmeans by C code,
 * and returns the matrix needed for python.
 */
static PyObject* getMatrixByGoal(PyObject *self, PyObject *args){
    PyObject *py_matrix;
    PyObject *final_mat;
    PyObject *eigens_list;
    int n, k, i;
    int vec_length;
    char* goal;
    double** original_matrix;
    double** w_matrix;
    double** d_matrix;
    double** l_matrix;
    double** t_matrix;
    struct eigens* eigens_arr;
    if (!PyArg_ParseTuple(args, "iiiOs",&k, &n, &vec_length, &py_matrix, &goal)){
        errorOccured();
    }

    original_matrix = makeMatrixFromPyInput(py_matrix, n, vec_length);
    final_mat = PyList_New(0);
    if (!final_mat)
        errorOccured();
    
    /* choose which goal to perform */
    if (strcmp(goal, "wam") == 0){
        w_matrix = wamCalc(original_matrix, n, vec_length);
        makeListFromMatrix(final_mat, w_matrix, n, n);

        freeMatrix(w_matrix, n);
    }

    else if (strcmp(goal, "ddg") == 0){
        w_matrix = wamCalc(original_matrix, n, vec_length);
        d_matrix = ddgCalc(w_matrix, n);
        makeListFromMatrix(final_mat, d_matrix, n, n);

        freeMatrix(w_matrix, n);
        freeMatrix(d_matrix, n);
    }

    else if (strcmp(goal, "lnorm") == 0){
        w_matrix = wamCalc(original_matrix, n, vec_length);
        d_matrix = ddgCalc(w_matrix, n);
        l_matrix = lnormCalc(w_matrix, d_matrix, n);
        makeListFromMatrix(final_mat, l_matrix, n, n);
        
        freeMatrix(w_matrix, n);
        freeMatrix(d_matrix, n);
        freeMatrix(l_matrix, n);
    }

    else if (strcmp(goal, "jacobi") == 0){
        double** jacobi_matrix;
        eigens_arr = jacobiCalc(original_matrix, n, 0);
        jacobi_matrix = jacobiMatForPrint(eigens_arr, n);
        eigens_list = PyList_New(0);
        if (!eigens_list){
            Py_DECREF(final_mat);
            errorOccured();
        }
        for (i=0; i<n; i++){
            PyList_Append(eigens_list, PyFloat_FromDouble((double)eigens_arr[i].value));
        }
        PyList_Append(final_mat, eigens_list);
        makeListFromMatrix(final_mat, jacobi_matrix, n, n);
        for (i = 0; i < n ; i++){
            free(eigens_arr[i].vector);
        }
        free(eigens_arr);
        freeMatrix(jacobi_matrix, n);
        
    }

    else if (strcmp(goal, "spk") == 0){
        w_matrix = wamCalc(original_matrix, n, vec_length);
        d_matrix = ddgCalc(w_matrix, n);
        l_matrix = lnormCalc(w_matrix, d_matrix, n);
        eigens_arr = jacobiCalc(l_matrix, n, 1);
        if (k == 0)
            k = eigengapHeuristic(eigens_arr, n);
        t_matrix = createMatrixT(eigens_arr, n, k);
        makeListFromMatrix(final_mat, t_matrix, n, k);

        for (i = 0; i < n ; i++){
            free(eigens_arr[i].vector);
        }
        freeMatrix(w_matrix, n);
        freeMatrix(d_matrix, n);
        freeMatrix(l_matrix, n);
        freeMatrix(t_matrix, n);
        free(eigens_arr);
    }
    
    freeMatrix(original_matrix, n);
    return Py_BuildValue("O", final_mat);
}

/**
 * function called from python.
 * performs the kmeans++ algorithm,
 * returns the centroids of the clusters.
 */
static PyObject* fit(PyObject *self, PyObject *args){
    PyObject *centroids_arr;
    PyObject *py_centroids;
    PyObject *py_vectors;
    int n, k, d, max_iter; 
    double** elements;
    double** centroids;

    if (!PyArg_ParseTuple(args, "iiiiOO", &k, &n, &d, &max_iter, &py_centroids, &py_vectors)){
        errorOccured();
    }
    elements = makeMatrixFromPyInput(py_vectors, n, d);
    centroids = makeMatrixFromPyInput(py_centroids, k, d);

    /* K-means operation done by spkmeans.c functions */
    getFinalCentroids(centroids, elements, k, d, n, max_iter, 0);
    centroids_arr = PyList_New(0);
    if (!centroids_arr)
        errorOccured();
    makeListFromMatrix(centroids_arr, centroids, k, k);

    freeMatrix(elements, n);
    freeMatrix(centroids, k);
    return Py_BuildValue("O", centroids_arr);
}


 /* Python C-API functions */
static PyMethodDef spkmeansmoduleMethods[] = {
        {"getMatrixByGoal", (PyCFunction) getMatrixByGoal, METH_VARARGS, PyDoc_STR
        ("C-api for python, retrieving the matrix accoring to the goal")},
        {"fit", (PyCFunction) fit, METH_VARARGS, PyDoc_STR("Kmeans")},
        
        {NULL,  NULL, 0, NULL}
};

static struct PyModuleDef _moduledef = {
        PyModuleDef_HEAD_INIT,
        "spkmeansmodule",
        NULL,
        -1,
        spkmeansmoduleMethods
};

PyMODINIT_FUNC PyInit_spkmeansmodule(void)
{
    PyObject *module;
    module = PyModule_Create(&_moduledef);
    if (!module){
        errorOccured();
    }
    return module;
}
