#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void cudaAddKernal(int* A, int* B, int* C);
cudaError_t CudaAdd(int* A, int* B, int* C, size_t size);

