#include "matrix_mul_naive.h"
#include <iostream>

__global__ void MultiplyNaive(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
{
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;



    if (col_index < bCols && row_index < aRows)
    {
        float sum = 0;
     
        for (int i = 0; i < inner; i++)
        {
            sum += A[row_index * inner + i] * B[col_index + (i * bCols)];
        }
     
        C[row_index * aRows + col_index] = sum;
     
    }
    else
        return; //thread is out of bound

}




float MatrixMultNaiveCuda(CudaMatMulHandle& context)
{
    //Timing stuff

    cudaEvent_t start, stop;
    //Measure elapsed time in Kernel in milliseconds
    float ms = 0.0f;
    //Create timing evetns
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
    dim3 blocksPerGrid( (context.bCols + BLOCKWIDTH - 1) / BLOCKWIDTH , ( (context.aRows + BLOCKWIDTH - 1) / BLOCKWIDTH) );

    cudaError_t status;


    status = cudaEventRecord(start, cudaEventRecordDefault);
    if (CheckError(status, "Start Event Record failed   -->", __FILE__, __LINE__) != 0)
        return -1;
    MultiplyNaive << <blocksPerGrid, threadsPerBlock >> > (
    context.dev_a,
    context.dev_b,
    context.dev_c,
    context.aRows,
    context.inner,
    context.bCols
    );

    status = cudaGetLastError();
    if (CheckError(status, "Get Last Error after Kernal Call Failure- -->", __FILE__, __LINE__) != 0)
        return -1;
    //Device Sync 
    status = cudaDeviceSynchronize();
    if (CheckError(status, "Syncronize failed -->", __FILE__, __LINE__) != 0)
        return -1;

    //Everything is finished so we can Get stop time
    status = cudaEventRecord(stop, cudaEventRecordDefault);
    if (status != cudaSuccess)
    {
        std::cerr << "Stop Event Record failed";
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