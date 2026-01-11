//Converts 2d Vector into 1d array
#include "cuda_utils.h"
#include <iostream>     //for std::cerr;
#include <random>


std::vector<float> flatten(const std::vector< std::vector<float> >& tdv)
{
    if (tdv.empty()) {
        std::cerr << "flatten(): input has no rows\n";
        return {};
    }

    if (tdv[0].empty()) {
        std::cerr << "flatten(): input has no columns\n";
        return {};
    }

    size_t rowCount = tdv.size(); //number of columns within 2d vector
    size_t colCount = tdv[0].size();


    //Rectangluar Check
    for (size_t i = 1; i < rowCount; ++i)
    {
        if (tdv[i].size() != colCount)
        {
            std::cerr << "Input matrix is not Rectangular";
            return {};
        }
    }


    std::vector<float> flat(rowCount * colCount);
    /*  std::cout << sizeof(flattened);*/

    size_t idx = 0;
    for (size_t i = 0; i < rowCount; ++i)
    {
        for (size_t j = 0; j < colCount; ++j)
        {
            flat[idx++] = tdv[i][j];
        }
    }





    return flat;

}
std::vector<float> generateMatrix(size_t rows, size_t cols, int seed)
{
    std::vector<float>  Matrix(rows*cols);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 10);

    for (size_t i = 0; i < rows*cols; ++i)
    {
        
         Matrix[i] = (float)dist(rng);
    }
    return Matrix;
}
bool validateAdd(const float* inputA, const float* inputB, const float* result, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        if (result[i] != (inputA[i] + inputB[i]))
        {
            std::cout << result[i] << " != " <<inputA[i] << " + "<< inputB[i] << std::endl;
            return false;
        }
            
    }
    return true;
}



bool validateMultiply(const float* inputA, const float* inputB, const float* inputC, size_t aRows, size_t inner, size_t bCols)
{
   //Slow multiply A and B and check if it is equal to C
    
    float sum = 0;
    //Rows of A
    for (int rows = 0; rows < aRows; ++rows)
    {
        //Cols of B
        for (int cols = 0; cols < bCols; ++cols)
        {
            //Inner number of nums within each row of A and Col of B 
            for (int N = 0; N < inner; ++N)
            {
                sum += inputA[rows * inner + N] * inputB[N * bCols + cols];
            }
            if ((int)inputC[rows * bCols + cols] != (int)sum)
                return false;
            sum = 0;

        }

    }
    return true;
}




cudaError_t createMatMulContext(CudaMatMulHandle& context, size_t aRows, size_t inner, size_t bCols)
{  
    cudaError_t status;
    context.dev_a = 0;
    context.dev_b = 0;
    context.dev_c = 0;
    context.aRows = aRows;
    context.inner = inner;
    context.bCols = bCols;
    //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&context.dev_a, (aRows * inner * sizeof(float)));
    CheckError(status, "Malloc Failed A", __FILE__, __LINE__);


    status = cudaMalloc((void**)&context.dev_b, (bCols * inner * sizeof(float)));
    CheckError(status, "Malloc Failed B", __FILE__, __LINE__);

    status = cudaMalloc((void**)&context.dev_c, (aRows * bCols * sizeof(float)));
    CheckError(status, "Malloc Failed C", __FILE__, __LINE__);

   



Error:
    return status;
}


/*
* //Allocated Memory on GPU (DEVICE)
    status = cudaMalloc((void**)&context.dev_a, (context.aRows * context.inner * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed A");


    status = cudaMalloc((void**)&context.dev_b, (context.bCols * context.inner * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed B");

    status = cudaMalloc((void**)&context.dev_c, (context.aRows * context.bCols * sizeof(float)));
    CudaStatusCheck(status, "Malloc Failed C");

    //Copy Memory from host to Device. A and B only
    status = cudaMemcpy(dev_a, A, aRows * inner * sizeof(float), cudaMemcpyHostToDevice);
    CudaStatusCheck(status, "H->D MemCpy Failed w / A");


    status = cudaMemcpy(dev_b, B, inner * bCols * sizeof(float), cudaMemcpyHostToDevice);
    CudaStatusCheck(status, "H->D MemCpy Failed w/ B");
*/
cudaError_t copyMatMalInputsToDevice(CudaMatMulHandle& context, const float* host_A, const float* host_B)
{
    cudaError_t status;
    //Copy Memory from host to Device. A and B only
   status = cudaMemcpy(context.dev_a, host_A, context.aRows * context.inner * sizeof(float), cudaMemcpyHostToDevice);
   CheckError(status, "H->D MemCpy Failed input A   --> ", __FILE__, __LINE__);

    status = cudaMemcpy(context.dev_b, host_B, context.inner * context.bCols * sizeof(float), cudaMemcpyHostToDevice);
   CheckError(status, "H->D MemCpy Failed input B  --> ", __FILE__, __LINE__);


    return status;
}

cudaError_t copyMatMulOutputToHost(CudaMatMulHandle& context,float* host_C)
{
    cudaError_t status;
    //Copy C back to Host

    status = cudaMemcpy(host_C, context.dev_c, context.bCols * context.aRows * sizeof(float), cudaMemcpyDeviceToHost);
    CheckError(status, "D->H cudaMemcpy Failed for resulting Matrix C  -->", __FILE__, __LINE__);


    return status;
}
void destroyMatMulContext(CudaMatMulHandle& context)
{
    cudaFree(context.dev_c);
    cudaFree(context.dev_a);
    cudaFree(context.dev_b);
    return;
}
int CheckError(cudaError_t status, const char* errorMsg, const char* file, int line)
{
    if(status != cudaSuccess)
    {
        std::cout<< errorMsg << " "<< cudaGetErrorString(status) << "\nIn file: " << file << "\nOn line number: " << line;
        return -1;
    }
    return 0;
}

