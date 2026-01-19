#pragma once
#include "cuda_utils.h"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <algorithm>

//V1 of reduction. Naive implementation
__global__ void reduce_v1(const float* input, float* output, size_t N);
float cudaReduce_v1(CudaReduceHandle& ctx);