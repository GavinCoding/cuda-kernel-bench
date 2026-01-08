#include "matrix_mul.h"

#include <iostream>


__global__ void matrixMultNaiveKernel(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
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
        if(j < bCols && ((t * TILESIZE + ty) < inner))
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

cudaError_t MatrixMultCuda(const float* A, const float* B, float* C, size_t aRows, size_t inner, size_t bCols)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 threadsPerBlock(TILESIZE, TILESIZE);
    dim3 blocksPerGrid(aRows* bCols / 2, aRows * bCols / 2);

    cudaError_t status;

    //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&dev_a, (aRows * inner * sizeof(float)));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed A";
        goto Error;
    }

    status = cudaMalloc((void**)&dev_b, (bCols * inner * sizeof(float)));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed B";
        goto Error;
    }

    status = cudaMalloc((void**)&dev_c, (aRows * bCols * sizeof(float)));
    if (status != cudaSuccess)
    {
        std::cerr << "Malloc Failed C";
        goto Error;
    }

    //Copy Memory from host to Device. A and B only
    status = cudaMemcpy(dev_a, A, aRows * inner * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "H->D MemCpy Failed w/ A";
        goto Error;
    }

    status = cudaMemcpy(dev_b, B, inner * bCols * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cerr << "H->D MemCpy Failed w/ B";
        goto Error;
    }

    //Call
    //Call
    //Call


   

    matMulTiled << <blocksPerGrid, threadsPerBlock >> > (
        dev_a,
        dev_b,
        dev_c,
        aRows,
        inner,
        bCols
        );

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
    status = cudaMemcpy(C, dev_c, bCols * aRows * sizeof(float), cudaMemcpyDeviceToHost);
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