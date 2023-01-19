#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int cols = mat -> cols;
    int index = row * cols + col;
    double data = mat -> data[index];
    return data;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int cols = mat -> cols;
    int index = row * cols + col;
    mat -> data[index] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if(rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix *new_mat = malloc(sizeof(matrix));
    if(new_mat == NULL) {
        return -2;
    }
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    double *data = malloc(sizeof(double) * (size_t)(rows * cols));
    if(data == NULL) {
        return -2;
    }

    for(int i = 0; i < rows * cols; i ++ ) {
        data[i] = 0.0;
    }
    new_mat->data = data;
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    new_mat->rows = rows;
    new_mat->cols = cols;
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    new_mat->parent = NULL;
    // 6. Set the `ref_cnt` field to 1.
    new_mat->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = new_mat;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if(mat == NULL) {
        return;
    }
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if(mat->parent == NULL) {
        mat->ref_cnt -- ;
    }

    if(mat->ref_cnt == 0) {
        free(mat);
    } else {
        deallocate_matrix(mat->parent);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (offset < 0 || rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix* new_mat = malloc(sizeof(matrix));
    if(new_mat == NULL) {
        return -2;
    }
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    new_mat->data = (from->data) + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    new_mat->rows = rows;
    new_mat->cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    new_mat->parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    from->ref_cnt ++ ;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = new_mat;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int rows = mat->rows;
    int cols = mat->cols;

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        // int step = ARRAY_SIZE / num_threads;
        int step = rows * cols / num_threads;
        int start = tid * step;
        int end = start + step;

        for(int i = 0; i < (rows * cols / 4 * 4); i += 4 ) {
            _mm256_storeu_pd (mat->data + i, _mm256_set1_pd(val));
        }
        // for(int i = start; i < end / 4 * 4; i += 4 ) {
        //     _mm256_storeu_pd (mat->data + i, _mm256_set1_pd(val));
        // }

        for(int i = (rows * cols / 4 * 4); i < rows * cols; i ++ ) {
            mat->data[i] = val;
        }

        // for(int i = end / 4 * 4; i < end; i ++ ) {
        //     mat->data[i] = val;
        // }
    }

    // for(int i = 0; i < rows; i ++ ) {
    //     for(int j = 0; j < cols; j ++ ) {
    //         printf("%lf ", mat->data[i * cols + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int rows = mat->rows;
    int cols = mat->cols;

    __m256d zero = _mm256_set1_pd(0);

    for(int i = 0; i < (rows * cols / 4 * 4); i += 4 ) {
        __m256d tmp = _mm256_loadu_pd(mat->data + i);
        __m256d mask = _mm256_cmp_pd(tmp, zero, 1);
        __m256d flex = _mm256_and_pd(tmp, mask);
        __m256d flop = _mm256_sub_pd(zero, flex);
        
        tmp = _mm256_max_pd(tmp, flop);

        _mm256_storeu_pd(result->data + i, tmp);
    }

    for(int i = rows * cols / 4 * 4; i < rows * cols; i ++ ) {
        if(mat->data[i] >= 0) {
            result->data[i] = mat->data[i];
        } else {
            result->data[i] = -1 * mat->data[i];
        }
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int rows = mat->rows;
    int cols = mat->cols;
    
    __m256d zero = _mm256_set1_pd(0);

    for(int i = 0; i < (rows * cols / 4 * 4); i += 4 ) {
        __m256d tmp = _mm256_loadu_pd (mat->data + i);
        _mm256_storeu_pd (result->data + i, _mm256_sub_pd(zero, tmp));
    }

    for(int i = rows * cols / 4 * 4; i < rows * cols; i ++ ) {
        result->data[i] = 0 - mat->data[i];
    }
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int rows = mat1->rows;
    int cols = mat1->cols;

    for(int i = 0; i < (rows * cols / 4 * 4); i += 4 ) {
        __m256d tmp_1 = _mm256_loadu_pd (mat1->data + i);
        __m256d tmp_2 = _mm256_loadu_pd (mat2->data + i);
        _mm256_storeu_pd (result->data + i, _mm256_add_pd(tmp_1, tmp_2));
    }

    for(int i = rows * cols / 4 * 4; i < rows * cols; i ++ ) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int rows = mat1->rows;
    int cols = mat1->cols;

    for(int i = 0; i < (rows * cols / 4 * 4); i += 4 ) {
        __m256d tmp_1 = _mm256_loadu_pd (mat1->data + i);
        __m256d tmp_2 = _mm256_loadu_pd (mat2->data + i);
        _mm256_storeu_pd (result->data + i, _mm256_sub_pd(tmp_1, tmp_2));
    }

    for(int i = rows * cols / 4 * 4; i < rows * cols; i ++ ) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    int row1 = mat1->rows;
    int col1 = mat1->cols;
    int col2 = mat2->cols;
    fill_matrix(result, 0);

    for(int i = 0; i < row1 / 4 * 4; i += 4) {
        for(int j = 0; j < col1 / 4 * 4; j += 4) {
            for(int k = 0; k < col2 / 4 * 4; k += 4) {
                __m256d tmp_1 = _mm256_loadu_pd (mat1->data + i * col1 + j);
                __m256d tmp_2 = _mm256_loadu_pd (mat2->data + j * col2 + k);
                __m256d tmp_c = _mm256_loadu_pd (result->data + i * col2 + k);
                _mm256_storeu_pd (result->data + i, _mm256_fmadd_pd (tmp_1, tmp_2, tmp_c));
            }
        }
    }

    for (int i = row1 / 4 * 4; i < row1; i ++ ) {
        for (int j = col1 / 4 * 4; j < col1; j ++ ) {
            for (int k = col2 / 4 * 4; k < col2; k ++ ) {
                result->data[i * col2 + k] += mat1->data[i * col1 + j] * mat2->data[j * col2 + k];
            }
        }
    }

    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int row = mat->rows;
    int col = mat->cols;
    if(pow == 0) {
        for(int i = 0; i < row; i ++ ) {
            for(int j = 0; j < col; j ++ ) {
                if(i != j) {
                    result->data[i * col + j] = 0;
                } else {
                    result->data[i * col + j] = 1;
                }
            }
        }
    } else {
        double* data = malloc(sizeof(double) * (size_t)(row * col));
        for(int i = 0; i < row * col; i ++ ) {
            data[i] = mat->data[i];
        }

        for(int power = 1; power < pow; power ++ ) {
            double* tmp = malloc(sizeof(double) * (size_t)(row * col));

            for(int i = 0; i < row * col; i ++ ) {
                tmp[i] = data[i];
            }

            matrix* tmp_mat = malloc(sizeof(matrix));
            tmp_mat->data = tmp;
            tmp_mat->rows = row;
            tmp_mat->cols = col;

            mul_matrix(result, mat, tmp_mat);

            for(int i = 0; i < row * col; i ++ ) {
                data[i] = result->data[i];
            }

            free(tmp_mat);
            free(tmp);
        }

        for(int i = 0; i < row; i ++ ) {
            for(int j = 0; j < col; j ++ ) {
                printf("%lf ", result->data[i * col + j]);
            }
            printf("\n");
        }

        printf("\n");

        free(data);
    }
    return 0;
}
