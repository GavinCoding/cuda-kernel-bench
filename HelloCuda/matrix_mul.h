#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define TILESIZE 20
/*
*  MATRIX MULTIPLE NAIVE KERNAL. Multiplies Matrixes represented as Row-Major flattened arrays.
*  A*B = C. numRows is how many Rows in C which is the same as the number of rows in A. numCols is the Number of Columns in C and its the same as the number of Cols in B
*  Size Represents the number of Cols in A and the number of Rows in B
*/
__global__ void matrixMultNaiveKernel(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols);

/*
*	MATRIX MULTIPLE Tiled KERNAL. Multiplies Matrixes represented as Row-Major flattened arrays.	A*B = C.
*	aRows is numbers of rows in A, inner is the number of Cols in A and number of Rows in b which must be equal. bCols is number of Cols in b.
*	Result C will have aRows rows and bCols cols
*/
__global__ void matMulTiled(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols);

cudaError_t MatrixMultCuda(const float* A, const float* B, float* C, size_t numRows, size_t numCols, size_t size);
