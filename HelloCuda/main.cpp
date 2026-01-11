#include "cuda_utils.h"

#include "matrix_add.h"
#include "matrix_mul_naive.h"
#include "matrix_mul_tiled.h"

#include <iomanip>

#include <vector>
#include <iostream>

int main() {

    const int aRows = 1243;
    const int aCols = 2354;
    const int bRows = 2354;
    const int bCols = 5254;

    //Generate Input Matrices
    std::vector<float> A = generateMatrix(aRows, aCols, 1);
    std::vector<float> B = generateMatrix(bRows, bCols, 2);

    //Make Result Matrix
    std::vector<float> C(aRows * bCols, 0);

    //Testing
    std::vector<float> NaiveRes;
    std::vector<float> TiledRes;
    int numSamples = 15;

    if (aCols != bRows)
    {
        std::cerr << "Matrices A and B are not Compatible for multiplication.";
    }
    
    float naiveSum = 0.0f;
    float tiledSum = 0.0f;

    float naiveMax = FLT_MIN;
    float tiledMax = FLT_MIN;

    float naiveMin = FLT_MAX;
    float tiledMin = FLT_MAX;

    cudaError_t status;

    CudaMatMulHandle context{};

    status = createMatMulContext(context, aRows, aCols, bCols);
    if (CheckError(status, "createMatMulContext Failed -> ", __FILE__, __LINE__) != 0)
        return -1;
         


    status = copyMatMalInputsToDevice(context, A.data(), B.data());
    if (CheckError(status, "copyMatMalInputsToDevice Failed ->", __FILE__, __LINE__) != 0)
    {
        return -1;
    }

    float naiveTemp = 0;
    float tiledTemp = 0;

    for (int i = 0; i < numSamples; i++)
    {
        naiveTemp = MatrixMultNaiveCuda(context);
        tiledTemp = MatrixMultTiledCuda(context);
        
        if (naiveTemp == -1 || tiledTemp == -1)
            break;

        NaiveRes.push_back(naiveTemp);
        TiledRes.push_back(tiledTemp);

        naiveSum += NaiveRes[i];
        tiledSum += TiledRes[i];

        //Max min checks
        if (NaiveRes[i] > naiveMax)
            naiveMax = NaiveRes[i];
        if(NaiveRes[i] < naiveMin)
            naiveMin = NaiveRes[i];


        if (TiledRes[i] > tiledMax)
            tiledMax = TiledRes[i];
        if (TiledRes[i] < tiledMin)
            tiledMin = TiledRes[i];
    }

    if (naiveTemp != -1 && tiledTemp != -1)
    {
        std::cout << "GPU-Time MatMul NAIVE  BEST: " << naiveMin << " AVERAGE: " << naiveSum / numSamples << " WORST: " << naiveMax << std::endl;

        std::cout << "GPU-Time MatMul TILED  BEST: " << tiledMin << " AVERAGE: " << tiledSum / numSamples << " WORST: " << tiledMax << std::endl;

        std::cout << (naiveSum) / (tiledSum) << "AVG       speedup from Tiled to Naive" << std::endl;
    }
    
    
 
  
Error:
   
    destroyMatMulContext(context);
    return 0;
}
