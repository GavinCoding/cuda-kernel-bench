#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define TILESIZE 2
/*
*  MATRIX MULTIPLE NAIVE KERNAL. Multiplies Matrixes repersented as Row-Major flattened arrays.
*  A*B = C. numRows is how many Rows in C which is the same as the number of rows in A. numCols is the Number of Columns in C and its the same as the number of Cols in B
*  Size Represents the number of Cols in A and the number of Rows in B
*/
__global__ void matrixMultNaiveKernel(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size);
__global__ void matMulTiled(const int* A, const int* B, int* C, size_t size);

cudaError_t MatrixMultCuda(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size);
