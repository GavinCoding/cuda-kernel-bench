#include "matrix_mul.h"

#include <iostream>


__global__ void matrixMultNaiveKernel(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size)
{
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;



    if (col_index < numCols && row_index < numRows)
    {
        int sum = 0;

        for (int i = 0; i < size; i++)
        {
            sum += A[row_index * size + i] * B[col_index + (i * numCols)];
        }

        C[row_index * numRows + col_index] = sum;




    }
    else
        return; //thread is out of bound




}

cudaError_t MatrixMultCuda(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    dim3 blockDim(16, 16);

    cudaError_t status;

    //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&dev_a, (numRows * size * sizeof(int)));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed A";
        goto Error;
    }

    status = cudaMalloc((void**)&dev_b, (size * numCols * sizeof(int)));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed B";
        goto Error;
    }

    status = cudaMalloc((void**)&dev_c, numRows * numCols * sizeof(int));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed C";
        goto Error;
    }

    //Copy Memory from host to Device. A and B only
    status = cudaMemcpy(dev_a, A, numRows * size * sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "H->D MemCpy Failed w/ A";
        goto Error;
    }

    status = cudaMemcpy(dev_b, B, size * numCols * sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "H->D MemCpy Failed w/ B";
        goto Error;
    }

    //Call
    //Call
    //Call


    matrixMultNaiveKernel << <16, blockDim >> > (dev_a, dev_b, dev_c, numRows, numCols, size);

    //Check for errors from Kernal Launch
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cerr << "Get Last Error after Kernal Call Failure";
        goto Error;
    }




    //Device Sync 
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cerr << "Syncronize failed";
        goto Error;
    }




    //Copy C back to Host
    status = cudaMemcpy(C, dev_c, numCols * numRows * sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        std::cerr << "D->H cudaMemcpy Failed for resulting Matrix C";
        goto Error;
    }


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return status;
}