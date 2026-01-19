#include "cuda_utils.h"

#include "matrix_add.h"
#include "matrix_mul_naive.h"
#include "matrix_mul_tiled.h"
#include "cublas_matMul.h"

#include "reduce.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cfloat>


//DEBUG
//nvcc -g -G -arch=sm_89 -Xcompiler="/Od /Zi" main.cu matrix_mul_naive.cu matrix_mul_tiled.cu cublas_matMul.cu reduce.cu cuda_utils.cu -lcublas -o cuda_app_debug.exe


//RELEASE
//nvcc -O3 -lineinfo -arch=sm_89 --use_fast_math main.cu matrix_mul_naive.cu matrix_mul_tiled.cu cublas_matMul.cu reduce.cu cuda_utils.cu -lcublas -o cuda_app.exe


//Then
// cuda_app matmul <M> <K> <N> <samples>
// or
// cuda_app reduce <num_elements>

// -----------------------------------------------------------------------------
// Data generation wrappers (intent-focused)
// -----------------------------------------------------------------------------

std::vector<float> generateVector(int n, float seed)
{
    // Internally reuses matrix generator, but semantically 1D
    return generateMatrix(1, n, seed);
}

std::vector<float> generateMatrix2D(int rows, int cols, float seed)
{
    return generateMatrix(rows, cols, seed);
}

// -----------------------------------------------------------------------------
// Reduction runner
// -----------------------------------------------------------------------------

int runReduction(int numElements, int numSamples)
{
    std::vector<float> input = generateVector(numElements, 1.0f);

    CudaReduceHandle context{};
    cudaError_t status;

    status = createReduceContext(context, numElements);
    if (CheckError(status, "createReduceContext Failed -> ", __FILE__, __LINE__) != 0)
        return -1;

    status = copyReduceInputsToDevice(context, input.data());
    if (CheckError(status, "copyReduceInputsToDevice Failed -> ", __FILE__, __LINE__) != 0)
        {destroyReduceContext(context);return 0;}

    float v1Sum = 0.0f, v2Sum = 0.0f;
    float v1Min = FLT_MAX, v2Min = FLT_MAX;
    float v1Max = FLT_MIN, v2Max = FLT_MIN;

    float result = 0.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        // --------------------
        // v1
        // --------------------
        cudaMemset(context.dev_output, 0, sizeof(float));

        float t1 = cudaReduce_v1(context);
        if (t1 == -1)
            {destroyReduceContext(context);return 0;}

        copyReduceOutputToHost(context, &result);


        v1Sum += t1;
        v1Min = std::min(v1Min, t1);
        v1Max = std::max(v1Max, t1);

        // --------------------
        // v2
        // --------------------
        cudaMemset(context.dev_output, 0, sizeof(float));

        float t2 = cudaReduce_v2(context);
        if (t2 == -1)
        {destroyReduceContext(context);return 0;}

        copyReduceOutputToHost(context, &result);

        v2Sum += t2;
        v2Min = std::min(v2Min, t2);
        v2Max = std::max(v2Max, t2);
    }

    constexpr int W_LABEL = 24;
    constexpr int W_NUM = 10;

    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << std::left << std::setw(W_LABEL) << "Kernel"
        << std::right << std::setw(W_NUM) << "BEST"
        << std::setw(W_NUM) << "AVG"
        << std::setw(W_NUM) << "WORST"
        << "\n";

    std::cout << std::string(44, '-') << "\n";

    std::cout
        << std::left << std::setw(W_LABEL) << "Reduction v1"
        << std::right << std::setw(W_NUM) << v1Min
        << std::setw(W_NUM) << (v1Sum / numSamples)
        << std::setw(W_NUM) << v1Max
        << "\n";

    std::cout
        << std::left << std::setw(W_LABEL) << "Reduction v2"
        << std::right << std::setw(W_NUM) << v2Min
        << std::setw(W_NUM) << (v2Sum / numSamples)
        << std::setw(W_NUM) << v2Max
        << "\n";

    destroyReduceContext(context);
    return 0;
}

// -----------------------------------------------------------------------------
// MatMul benchmark runner
// -----------------------------------------------------------------------------

int runMatMulBench(int aRows, int aCols, int bCols, int numSamples)
{
    const int bRows = aCols;

    std::vector<float> A = generateMatrix2D(aRows, aCols, 1.0f);
    std::vector<float> B = generateMatrix2D(bRows, bCols, 2.0f);

    CudaMatMulHandle context{};
    cudaError_t status;

    status = createMatMulContext(context, aRows, aCols, bCols);
    if (CheckError(status, "createMatMulContext Failed -> ", __FILE__, __LINE__) != 0)
        return -1;

    status = copyMatMalInputsToDevice(context, A.data(), B.data());
    if (CheckError(status, "copyMatMalInputsToDevice Failed -> ", __FILE__, __LINE__) != 0)
        {destroyMatMulContext(context);return 0;}

    float naiveSum = 0.0f, tiledSum = 0.0f, cublasSum = 0.0f;
    float naiveMin = FLT_MAX, tiledMin = FLT_MAX, cublasMin = FLT_MAX;
    float naiveMax = FLT_MIN, tiledMax = FLT_MIN, cublasMax = FLT_MIN;

    for (int i = 0; i < numSamples; i++)
    {
        float n = MatrixMultNaiveCuda(context);
        float t = MatrixMultTiledCuda(context);
        float c = MatMulCUblas(context);

        if (n == -1 || t == -1 || c == -1)
            break;

        naiveSum += n; tiledSum += t; cublasSum += c;

        naiveMin = std::min(naiveMin, n);
        naiveMax = std::max(naiveMax, n);

        tiledMin = std::min(tiledMin, t);
        tiledMax = std::max(tiledMax, t);

        cublasMin = std::min(cublasMin, c);
        cublasMax = std::max(cublasMax, c);
    }

    constexpr int W_LABEL = 24;
    constexpr int W_NUM = 10;

    std::cout << std::fixed << std::setprecision(3);
    std::cout
        << std::left << std::setw(W_LABEL) << "Kernel"
        << std::right << std::setw(W_NUM) << "BEST"
        << std::setw(W_NUM) << "AVG"
        << std::setw(W_NUM) << "WORST"
        << std::setw(W_NUM + 6) << "% cuBLAS\n";

    std::cout << std::string(60, '-') << "\n";

    std::cout
        << std::left << std::setw(W_LABEL) << "Naive"
        << std::right << std::setw(W_NUM) << naiveMin
        << std::setw(W_NUM) << naiveSum / numSamples
        << std::setw(W_NUM) << naiveMax
        << std::setw(W_NUM + 6) << (cublasSum / naiveSum) * 100.0f
        << "\n";

    std::cout
        << std::left << std::setw(W_LABEL) << "Tiled"
        << std::right << std::setw(W_NUM) << tiledMin
        << std::setw(W_NUM) << tiledSum / numSamples
        << std::setw(W_NUM) << tiledMax
        << std::setw(W_NUM + 6) << (cublasSum / tiledSum) * 100.0f
        << "\n";

    std::cout
        << std::left << std::setw(W_LABEL) << "cuBLAS"
        << std::right << std::setw(W_NUM) << cublasMin
        << std::setw(W_NUM) << cublasSum / numSamples
        << std::setw(W_NUM) << cublasMax
        << std::setw(W_NUM + 6) << "100.000\n";

    destroyMatMulContext(context);
    return 0;
}

// -----------------------------------------------------------------------------
// main: parse -> dispatch -> exit
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage:\n";
        std::cout << "  app reduce <num_elements>\n";
        std::cout << "  app matmul <M> <K> <N> <samples>\n";
        return 0;
    }

    if (std::strcmp(argv[1], "reduce") == 0)
    {
        int n = (argc >= 3) ? std::atoi(argv[2]) : 1024;
        int samples = (argc >= 4) ? std::atoi(argv[3]) : 10;
        return runReduction(n, samples);
    }

    if (std::strcmp(argv[1], "matmul") == 0)
    {
        if (argc < 6)
        {
            std::cerr << "matmul requires M K N samples\n";
            return -1;
        }

        int M = std::atoi(argv[2]);
        int K = std::atoi(argv[3]);
        int N = std::atoi(argv[4]);
        int samples = std::atoi(argv[5]);

        return runMatMulBench(M, K, N, samples);
    }

    std::cerr << "Unknown mode: " << argv[1] << "\n";
    return -1;
}