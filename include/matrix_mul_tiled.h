#pragma once
#include "cuda_utils.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define TILESIZE 32


/*
*	MATRIX MULTIPLE Tiled KERNAL. Multiplies Matrixes represented as Row-Major flattened arrays.	A*B = C.
*	aRows is numbers of rows in A, inner is the number of Cols in A and number of Rows in b which must be equal. bCols is number of Cols in b.
*	Result C will have aRows rows and bCols cols
*/
__global__ void matMulTiled(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols);
float MatrixMultTiledCuda(CudaMatMulHandle& context);