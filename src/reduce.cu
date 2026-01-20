#include "reduce.h"
#include <iostream>

__global__ void reduce_v1(const float* input, float* output, size_t N)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	float sum = 0.0f;
	for (size_t i = idx; i < N ;i += stride)
	{
		sum += input[i];
	}
	atomicAdd(output, sum);
}
__global__ void reduce_v2(const float* input, float* blockSums, size_t N)
{
	//Shared memory per block. Dynamic since it's based on block size
	extern __shared__ float sdata[];


	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	//each thread loads into shared memory then we sync
	sdata[tid] = (idx < N)  ? input[idx] : 0.0f;
	__syncthreads();
	
	//Now all a block of threads amount of data is loaded into shared memory. We add up all values in shared memory towards sdata[0]. Thread 0 is responsible for writing to block sums array
	for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{

		if(tid < stride)
		{
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
	{
		blockSums[blockIdx.x] = sdata[0];
	}

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

	int maxBlocks = 1024;
	dim3 blockSize(BLOCKSIZE);

	dim3 gridSize = std::min(maxBlocks, (int)((ctx.N + blockSize.x - 1) / blockSize.x));
	

	reduce_v1<<< gridSize, blockSize>>>(ctx.dev_input, ctx.dev_output, ctx.N);

	status = cudaGetLastError();
	if (CheckError(status, "Get Last Error after Kernal Call Failure   ->", __FILE__, __LINE__) != 0)
		return -1;
	//Device Sync 
	/*status = cudaDeviceSynchronize();
	if (CheckError(status, "Syncronize failed   ->", __FILE__, __LINE__) != 0)
		return -1;*/
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
float cudaReduce_v2(CudaReduceHandle& ctx)
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

	dim3 blockSize(BLOCKSIZE);

	int blocksNeeded = ctx.numBlocks;
	dim3 gridSize = blocksNeeded;

	reduce_v2 << < gridSize, blockSize, blockSize.x * sizeof(float) >> > (ctx.dev_input, ctx.dev_blockSums, ctx.N);

	status = cudaDeviceSynchronize();
	if (CheckError(status, "v2 Syncronize failed   ->", __FILE__, __LINE__) != 0)
		return -1;
	

	//Secondary sumup
	std::vector<float> h_blockSums(blocksNeeded);
	cudaMemcpy(
		h_blockSums.data(),
		ctx.dev_blockSums,
		ctx.numBlocks * sizeof(float),
		cudaMemcpyDeviceToHost
	);

	float total = 0.0f;
	for (int i = 0; i < blocksNeeded; ++i)
		total += h_blockSums[i];

	cudaMemcpy(ctx.dev_output, &total, sizeof(float), cudaMemcpyHostToDevice);


	//Device Sync 
	status = cudaDeviceSynchronize();
	if (CheckError(status, "v1 Syncronize failed   ->", __FILE__, __LINE__) != 0)
		return -1;

	status = cudaGetLastError();
	if (CheckError(status, "Get Last Error after Kernal Call Failure   ->", __FILE__, __LINE__) != 0)
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
