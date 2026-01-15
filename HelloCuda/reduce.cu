#include "reduce.h"
#include <iostream>

__global__ void reduce_v1(const float* input, float* output, size_t N)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	float sum = 0.0f;
	for (int i = idx; i < N ;i += stride)
	{
		sum += input[i];
	}
	atomicAdd(output, sum);
}
float cudaReduce_v1(CudaReduceHandle& ctx)
{
	//Timing stuff
	cudaEvent_t start, stop;
	//Create timing event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//Measure elapsed time in Kernel in milliseconds
	float ms = 0.0f;


	cudaError_t status;
	status = cudaEventRecord(start, cudaEventRecordDefault);
	if (CheckError(status, "EventRecord Failed   ->", __FILE__, __LINE__) != 0)
		return -1;

	dim3 blockSize (256);
	dim3 gridSize((ctx.N + blockSize.x - 1) / blockSize.x);

	reduce_v1<<< gridSize, blockSize>>>(ctx.dev_input, ctx.dev_output, ctx.N);

	status = cudaGetLastError();
	if (CheckError(status, "Get Last Error after Kernal Call Failure   ->", __FILE__, __LINE__) != 0)
		return -1;
	//Device Sync 
	status = cudaDeviceSynchronize();
	if (CheckError(status, "Syncronize failed   ->", __FILE__, __LINE__) != 0)
		return -1;
	//Eveyrthing is finished so we can Get stop time
	status = cudaEventRecord(stop, cudaEventRecordDefault);
	if (status != cudaSuccess)
	{
		std::cerr << "Stop Event Record failed   ->";
		//goto Error;
	}
	else
	{
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
	}


	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	return ms;
}
