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
    CudaStatusCheck(status, "createMatMulContext Failed ->");


    status = copyMatMalInputsToDevice(context, A.data(), B.data());
    CudaStatusCheck(status, "copyMatMalInputsToDevice Failed ->");

    for (int i = 0; i < numSamples; i++)
    {


        NaiveRes.push_back(MatrixMultNaiveCuda(context));
        TiledRes.push_back(MatrixMultTiledCuda(context));

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

   
    std::cout << "GPU-Time MatMul NAIVE  BEST: " << naiveMin << " AVERAGE: " << naiveSum / numSamples << " WORST: " << naiveMax << std::endl;

    std::cout << "GPU-Time MatMul TILED  BEST: " << tiledMin << " AVERAGE: " << tiledSum / numSamples << " WORST: " <<tiledMax << std::endl;

    std::cout << (naiveSum) / (tiledSum) << "AVG       speedup from Tiled to Naive" << std::endl;;
    
 
  
Error:
   
    destroyMatMulContext(context);
    return 0;
}
