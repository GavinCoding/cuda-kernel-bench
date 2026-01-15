#include "cuda_utils.h"

#include "matrix_add.h"
#include "matrix_mul_naive.h"
#include "matrix_mul_tiled.h"
#include "cublas_matMul.h"

#include "reduce.h"

#include <iomanip>

#include <vector>
#include <iostream>




int main() {

    const int aRows = 3;
    const int aCols = 3;
    const int bRows = 3;
    const int bCols = 3;

    //Generate Input Matrices
    std::vector<float> A = generateMatrix(aRows, aCols, 1);
    std::vector<float> B = generateMatrix(bRows, bCols, 2);

    //Make Result Matrix
    std::vector<float> C(aRows * bCols, 0);

    //Testing
    std::vector<float> NaiveRes;
    std::vector<float> TiledRes;
    std::vector<float> CublasRes;
    int numSamples = 15;

    if (aCols != bRows)
    {
        std::cerr << "Matrices A and B are not Compatible for multiplication.";
    }

    float naiveSum = 0.0f;
    float tiledSum = 0.0f;
    float cublasSum = 0.0f;

    float naiveMax = FLT_MIN;
    float tiledMax = FLT_MIN;
    float cublasMax = FLT_MIN;

    float naiveMin = FLT_MAX;
    float tiledMin = FLT_MAX;
    float cublasMin = FLT_MAX;



    for (float num : A)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    cudaError_t status;

    CudaReduceHandle context{};

    status = createReduceContext(context, aRows * aCols);
    if (CheckError(status, "createReduceContext Failed -> ", __FILE__, __LINE__) != 0)
        return -1;

    status = copyReduceInputsToDevice(context, A.data());
    if (CheckError(status, "copyReduceInputsToDevice Failed ->", __FILE__, __LINE__) != 0)
        return -1;

    cudaMemset(context.dev_output, 0, sizeof(float));

    float result;
    float reduceKernelTime = cudaReduce_v1(context);
    

    if (reduceKernelTime == -1)
        goto Error;
    else
    {
        copyReduceOutputToHost(context, &result);
        std::cout << "reduced succeeded in: " << reduceKernelTime << " With Result: " << result;
    }
        
    Error:
    destroyReduceContext(context);
    return 0;
    //CudaMatMulHandle context{};
//    status = createMatMulContext(context, aRows, aCols, bCols);
//    if (CheckError(status, "createMatMulContext Failed -> ", __FILE__, __LINE__) != 0)
//        return -1;
//
//
//
//    status = copyMatMalInputsToDevice(context, A.data(), B.data());
//    if (CheckError(status, "copyMatMalInputsToDevice Failed ->", __FILE__, __LINE__) != 0)
//    {
//        return -1;
//    }
//
//    float naiveTemp = 0;
//    float tiledTemp = 0;
//    float cublasTemp = 0;
//
//    for (int i = 0; i < numSamples; i++)
//    {
//
//        naiveTemp = MatrixMultNaiveCuda(context);
//        tiledTemp = MatrixMultTiledCuda(context);
//        cublasTemp = MatMulCUblas(context);
//
//        if (naiveTemp == -1 || tiledTemp == -1 || cublasTemp == -1)
//            break;
//
//        NaiveRes.push_back(naiveTemp);
//        TiledRes.push_back(tiledTemp);
//        CublasRes.push_back(cublasTemp);
//        
//
//        naiveSum += naiveTemp;
//        tiledSum += tiledTemp;
//        cublasSum += cublasTemp;
//
//        naiveMax = std::max(naiveMax, naiveTemp);
//        naiveMin = std::min(naiveMin, naiveTemp);
//
//        tiledMax = std::max(tiledMax, tiledTemp);
//        tiledMin = std::min(tiledMin, tiledTemp);
//
//        
//
//        cublasMin = std::min(cublasMin, cublasTemp);
//        cublasMax = std::max(cublasMax, cublasTemp);
//    }
//
//
//    // formatting controls
//    constexpr int W_LABEL = 24;
//    constexpr int W_NUM = 10;
//
//    std::cout << std::fixed << std::setprecision(3);
//
//    if (naiveTemp != -1 && tiledTemp != -1 && cublasTemp != -1)
//    {
//        std::cout
//            << std::left << std::setw(W_LABEL) << "Kernel"
//            << std::right << std::setw(W_NUM) << "BEST(ms)"
//            << std::setw(W_NUM) << "AVG(ms)"
//            << std::setw(W_NUM) << "WORST(ms)"
//            << std::setw(W_NUM + 6) << "% of cuBLAS"
//            << "\n";
//
//        std::cout << std::string(60, '-') << "\n";
//
//        std::cout
//            << std::left << std::setw(W_LABEL) << "Naive"
//            << std::right << std::setw(W_NUM) << naiveMin
//            << std::setw(W_NUM) << (naiveSum / numSamples)
//            << std::setw(W_NUM) << naiveMax
//            << std::setw(W_NUM + 6) << (cublasSum / naiveSum) * 100.0f
//            << "\n";
//
//        std::cout
//            << std::left << std::setw(W_LABEL) << "Tiled"
//            << std::right << std::setw(W_NUM) << tiledMin
//            << std::setw(W_NUM) << (tiledSum / numSamples)
//            << std::setw(W_NUM) << tiledMax
//            << std::setw(W_NUM + 6) << (cublasSum / tiledSum) * 100.0f
//            << "\n";
//
//        std::cout
//            << std::left << std::setw(W_LABEL) << "cuBLAS SGEMM"
//            << std::right << std::setw(W_NUM) << cublasMin
//            << std::setw(W_NUM) << (cublasSum / numSamples)
//            << std::setw(W_NUM) << cublasMax
//            << std::setw(W_NUM + 6) << "100.000"
//            << "\n";
//
//        std::cout << "\n"
//            << "Avg speedup (Tiled vs Naive): "
//            << (naiveSum / tiledSum) << "x\n";
//    }
//   
//    
// 
//  
//Error:
//   
//    destroyMatMulContext(context);
//    return 0;
}
