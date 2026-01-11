#include "matrix_mul_tiled.h"
#include <iostream>

__global__ void matMulTiled(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
{
    //Indexing variables
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;


    //Calulate position of thread relative to data
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;



    //Split data into tiles
    //Tile Size is block size
    //We are storing parts of matrix A and B that are needed for calulation
    __shared__ float aShared[TILESIZE][TILESIZE];
    __shared__ float bShared[TILESIZE][TILESIZE];

    float value = 0;
    //int solSize;

    //load tiles into shared memory
    for (int t = 0; t < ceil((float)inner / TILESIZE); t++)
    {
        //Load tile from global after bounds checking
        if ((i < aRows) && ((t * TILESIZE + tx) < inner))
            aShared[ty][tx] = A[(i * inner) + (t * TILESIZE) + tx];
        else
            aShared[ty][tx] = 0.0f;
        //
        if (j < bCols && ((t * TILESIZE + ty) < inner))
            bShared[ty][tx] = B[((t * TILESIZE + ty) * bCols) + j];
        else
            bShared[ty][tx] = 0.0f;



        //sync so that relevant tiles are complete
        __syncthreads();

        //dot product of a and b tiles
        for (int j = 0; j < TILESIZE; j++)
        {
            value += aShared[ty][j] * bShared[j][tx];
        }
        __syncthreads();
        //inside K
    }
    if ((i < aRows) && (j < bCols))
    {
        C[i * bCols + j] = value;
    }
}

float MatrixMultTiledCuda(CudaMatMulHandle& context)
{
    //Timing stuff
    cudaEvent_t start, stop;
    //Create timing event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //Measure elapsed time in Kernel in milliseconds
    float ms = 0.0f;

    //dim stuff
    dim3 threadsPerBlock(TILESIZE, TILESIZE, 1);
    dim3 blocksPerGrid((context.bCols + TILESIZE - 1) / TILESIZE, ((context.aRows + TILESIZE - 1) / TILESIZE));

    

    cudaError_t status;
    status = cudaEventRecord(start, cudaEventRecordDefault);
    if(CheckError(status, "EventRecord Failed   ->", __FILE__, __LINE__) != 0)
        return -1;
   
    matMulTiled << <blocksPerGrid, threadsPerBlock >> > (
        context.dev_a,
        context.dev_b,
        context.dev_c,
        context.aRows,
        context.inner,
        context.bCols
        );

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