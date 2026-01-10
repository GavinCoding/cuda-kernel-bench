#include "matrix_mul_naive.h"
#include <iostream>

#define CudaStatusCheck(Status,Msg) {if(Status != cudaSuccess){std::cout<<Msg << " "<< cudaGetErrorString(Status) << "\nIn file: " << __FILE__ << "\nOn line number: " << __LINE__;goto Error;}}

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




float MatrixMultNaiveCuda(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
{
    //Timing stuff

    cudaEvent_t start, stop;
    //Measure elapsed time in Kernel in milliseconds
    float ms = 0.0f;
    //Create timing evetns
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 threadsPerBlock(BLOCKWIDTH, BLOCKWIDTH, 1);
    dim3 blocksPerGrid(ceil(bCols / (float)BLOCKWIDTH), ceil(aRows / (float)BLOCKWIDTH), 1);
    //dim3 blocksPerGrid( (bCols + BLOCKWIDTH - 1) / BLOCKWIDTH , ( (aRows + BLOCKWIDTH - 1) / BLOCKWIDTH) );

    cudaError_t status;

    //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&dev_a, (aRows * inner * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed A");


    status = cudaMalloc((void**)&dev_b, (bCols * inner * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed B");

    status = cudaMalloc((void**)&dev_c, (aRows * bCols * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed C");

    //Copy Memory from host to Device. A and B only
    status = cudaMemcpy(dev_a, A, aRows * inner * sizeof(float), cudaMemcpyHostToDevice);
    CudaStatusCheck(status, "H->D MemCpy Failed w / A");


    status = cudaMemcpy(dev_b, B, inner * bCols * sizeof(float), cudaMemcpyHostToDevice);
    CudaStatusCheck(status, "H->D MemCpy Failed w/ B");


    status = cudaEventRecord(start, cudaEventRecordDefault);
    CudaStatusCheck(status, "Start Event Record failed");

    MultiplyNaive << <blocksPerGrid, threadsPerBlock >> > (
    dev_a,
    dev_b,
    dev_c,
    aRows,
    inner,
    bCols
    );

    status = cudaGetLastError();
    CudaStatusCheck(status, "Get Last Error after Kernal Call Failure");

    //Device Sync 
    status = cudaDeviceSynchronize();
    CudaStatusCheck(status, "Syncronize failed");

    //Eveyrthing is finished so we can Get stop time
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


    //Copy C back to Host
    status = cudaMemcpy(C, dev_c, bCols * aRows * sizeof(float), cudaMemcpyDeviceToHost);
    CudaStatusCheck(status, "D->H cudaMemcpy Failed for resulting Matrix C");

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}