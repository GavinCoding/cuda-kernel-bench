#include "matrix_mul_tiled.h"
#include <iostream>

#define CudaStatusCheck(Status,Msg) {if(Status != cudaSuccess){std::cout<<Msg << " "<< cudaGetErrorString(Status) << "\nIn file: " << __FILE__ << "\nOn line number: " << __LINE__;goto Error;}}

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

float MatrixMultTiledCuda(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
{
    //Timing stuff
    cudaEvent_t start, stop;
    //Create timing event
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Measure elapsed time in Kernel in milliseconds
    float ms = 0.0f;

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;


    dim3 threadsPerBlock(TILESIZE, TILESIZE, 1);
    dim3 blocksPerGrid(ceil(bCols / (float)TILESIZE), ceil(aRows / (float)TILESIZE), 1);
    /*dim3 threadsPerBlock(TILESIZE, TILESIZE);

    dim3 blocksPerGrid((bCols + TILESIZE - 1) / TILESIZE, ((aRows + TILESIZE - 1) / TILESIZE));*/

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
   
    matMulTiled << <blocksPerGrid, threadsPerBlock >> > (
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