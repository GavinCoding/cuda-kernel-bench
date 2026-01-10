#include "cuda_utils.h"


#include "matrix_add.h"
#include "matrix_mul_naive.h"
#include "matrix_mul_tiled.h"

#include <iomanip>

#include <vector>
#include <iostream>

int main() {

    const int aRows = 1256;
    const int aCols = 3567;
    const int bRows = 3567;
    const int bCols = 2894;

    //Generate Input Matrices
    std::vector<float> A = generateMatrix(aRows, aCols, 1);
    std::vector<float> B = generateMatrix(bRows, bCols, 2);

    //Make Result Matrix
    std::vector<float> C(aRows * bCols, 0);

    //Testing
    std::vector<float> NaiveRes;
    std::vector<float> TiledRes;
    int numSamples = 5;

    if (aCols != bRows)
    {
        std::cerr << "Matrices A and B are not Compatible for multiplication.";
    }
    
    float naiveSum = 0.0f;
    float tiledSum = 0.0f;

    for (int i = 0; i < numSamples; i++)
    {
        NaiveRes.push_back(MatrixMultNaiveCuda(
            A.data(),
            B.data(),
            C.data(),
            aRows,
            aCols,
            bCols
        ));
        

        TiledRes.push_back(MatrixMultTiledCuda(
            A.data(),
            B.data(),
            C.data(),
            aRows,
            aCols,
            bCols
        ));

        naiveSum += NaiveRes[i];
        tiledSum += TiledRes[i];
    }
    
    std::cout << "AVG. GPU-Time MatMul      NAIVE:  " << naiveSum / numSamples << std::endl;
    std::cout << "AVG. GPU-Time MatMul      TILED:  " << tiledSum / numSamples << std::endl;

    std::cout << (naiveSum) / (tiledSum) << "x speedup from Tiled to Naive" << std::endl;;
    
 
   

  
    return 0;
}
