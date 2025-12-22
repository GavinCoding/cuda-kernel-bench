#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>


std::vector< std::vector<int> > multiply(const std::vector<std::vector<int>> A, const std::vector<std::vector<int>> B);
//Converts 2d Vector into 1d array 
std::vector<int> flatten(std::vector< std::vector<int> > tdv);


cudaError_t MatrixMultCuda(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size);
//A*B = C. numRows is how many Rows in C which is the same as the number of rows in A. numCols is the Number of Columns in C and its the same as the number of Cols in B
//Size Represents the number of Cols in A and the number of Rows in B
__global__ void matrixMultNaiveKernel(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size)
{
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;



    if (col_index < numCols && row_index < numRows)
    {
        int sum = 0;
        
        for (int i = 0; i < size; i++)
        {   
            sum += A[row_index*size + i] * B[col_index + (i*numCols)];
        }
        
        C[row_index* numRows + col_index] = sum;


        

    }
    else
        return; //thread is out of bound

   

    
}

int main() {
    std::vector< std::vector<int> > A =
    {
        {1,2,3},
        {4,5,6},
        {7,8,9},
        {10,11,12},
        {13,14,15}
    };

    std::vector< std::vector<int> > B =
    {
        {4,6,2,10,12},
        {1,4,5,4,2},
        {3,6,7,12,3}
    };

    std::vector<int> A_flat = flatten(A);
    std::vector<int> B_flat = flatten(B);

    int cSize = A.size() * B[0].size();
    int* C = new int[cSize];


    
    MatrixMultCuda(A_flat.data(), B_flat.data(), C, A.size(), B[0].size(), A[0].size());

    for (int i = 0; i < cSize; i++)
    {
        std::cout << C[i] << " ";
        if ((i + 1) % (B[0].size()) == 0)
           std::cout << std::endl;
    }
    delete[] C;
    return 0;

}
//Converts 2d Vector into 1d array 
std::vector<int> flatten (std::vector< std::vector<int> > tdv)
{
   
    size_t rowCount = tdv.size(); //number of columns within 2d vector
    size_t colCount = tdv[0].size();

   

    std::vector<int> flat;
  /*  std::cout << sizeof(flattened);*/
    
    for (size_t i = 0; i < rowCount; i++)
    {
        for (size_t j = 0; j < colCount; j++)
        {
            //std::cout << i * rowCount + j << " ";
            flat.push_back(tdv[i][j]);
        }
        //std::cout << std::endl;
    }





    return flat;
    
}
cudaError_t MatrixMultCuda(const int* A, const int* B, int* C, size_t numRows, size_t numCols, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;


    cudaError_t status;

    //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&dev_a, (numRows * size * sizeof(int)) );
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
    if(status != cudaSuccess)
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
    dim3 blockDim(16, 16);

    matrixMultNaiveKernel <<<16, blockDim >>> (dev_a, dev_b, dev_c, numRows, numCols, size);

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
    status = cudaMemcpy(C, dev_c, numCols * numRows *sizeof(int), cudaMemcpyDeviceToHost);
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
//cudaError_t MultiWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
//
//
//
//
std::vector< std::vector<int> > multiply(const std::vector<std::vector<int>> A, const std::vector<std::vector<int>> B)
{
    //Imagine we 
    //For Results
    size_t rowA = A.size();
    size_t colB = B[0].size();

    size_t colA = A[0].size();
    size_t rowB = B.size();


    //For Result
    std::vector< std::vector<int> > C(rowA, std::vector<int>(colB, 0));

    
    if (colA == rowB)
    {
        for (int i = 0; i < rowA; i++)
        {
            for (int j = 0; j < colB; j++)
            {
                for (int k = 0; k < rowB; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

    }
    else
    {
        std::cerr << "Can't Only multiple Maturx of size (nCmR) * (mCoR) ";
    }
        
    for(int RowNum = 0; RowNum < rowA; RowNum++)
    {
        for(int ColNum = 0; ColNum < colB; ColNum++)
        {
            std::cout << C[RowNum][ColNum] << " ";
        }
        std::cout << "\n";
    }

    return C;
}