#include "matrix_add.h"
#include <iostream>
#include <cmath>

__global__ void cudaAddKernal (int* A, int* B, int* C, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		C[idx] = A[idx] + B[idx];
	}
		

}
cudaError_t CudaAdd(int* A, int* B, int* C, size_t size)
{
	// Memory locations for matrices on device
	int* dev_A = 0;
	int* dev_B = 0;
	int* dev_C = 0;

	//Dimision for Kernal Call
	dim3 block(16, 16);
	/*dim3 grid()*/

	// Return code singleton for error checking
	cudaError_t results = cudaSuccess;


	//Init Memory on device 
	//A
	results = cudaMalloc((void**)&dev_A, size * sizeof(int));
	if (results != cudaSuccess)
	{
		std::cerr << "cuda Malloc failed on A: Errror code: " << results;
		goto Error;
	}

	results = cudaMalloc((void**)&dev_B, size * sizeof(int));
	if (results != cudaSuccess)
	{
		std::cerr << "cuda Malloc failed on B: Errror code: " << results;
		goto Error;
	}

	results = cudaMalloc((void**)&dev_C, size * sizeof(int));
	if (results != cudaSuccess)
	{
		std::cerr << "cuda Malloc failed on C: Error Code: " << results;
		goto Error;
	}

	//Next we Memcopy from Host to Memory
	results = cudaMemcpy((void*)dev_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
	if (results != cudaSuccess)
	{
		std::cerr << "MemCpy failed on A-> dev_A: Errror code: " << results;
		goto Error;
	}

	results = cudaMemcpy((void*)dev_B, B, size * sizeof(int), cudaMemcpyHostToDevice);
	if (results != cudaSuccess)
	{
		std::cerr << "MemCpy failed on B-> dev_B: Errror code: " << results;
		goto Error;
	}


	//Call Kernel with 16 blocks and 16 threads per block. 
	cudaAddKernal << <16, block >> > (dev_A, dev_B, dev_C, size);

	//Check for errors from Kernal Launch
	results = cudaGetLastError();
	if (results != cudaSuccess)
	{
		std::cerr << "Get Last Error after Kernal Call Failure";
		goto Error;
	}


	//Device Sync 
	results = cudaDeviceSynchronize();
	if (results != cudaSuccess)
	{
		std::cerr << "Syncronize failed";
		goto Error;
	}

	//Copy Results from Dev_C to Host C
	results = cudaMemcpy(C, dev_C, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (results != cudaSuccess)
	{
		std::cerr << "MemCpy failed on dev_C-> C: Errror code: " << results;
		goto Error;
	}
	


Error:
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	return results;
		
}