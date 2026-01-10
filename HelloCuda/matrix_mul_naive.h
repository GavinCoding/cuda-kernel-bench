#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#define BLOCKWIDTH 32
/*
*  MATRIX MULTIPLE NAIVE KERNAL. Multiplies Matrixes represented as Row-Major flattened arrays.
*  A*B = C. numRows is how many Rows in C which is the same as the number of rows in A. numCols is the Number of Columns in C and its the same as the number of Cols in B
*  Size Represents the number of Cols in A and the number of Rows in B
*/
__global__ void MultiplyNaive(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols);
float MatrixMultNaiveCuda(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols);