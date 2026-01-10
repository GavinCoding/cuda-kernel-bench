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

    //Result times
    float NaiveRes = 0.0f;
    float TiledRes = 0.0f;

    if (aCols != bRows)
    {
        std::cerr << "Matrices A and B are not Compatible for multiplication.";
    }
    
    NaiveRes = MatrixMultNaiveCuda(
        A.data(),
        B.data(),
        C.data(),
        aRows,
        aCols,
        bCols
    );
    

    TiledRes = MatrixMultTiledCuda(
        A.data(),
        B.data(),
        C.data(),
        aRows,
        aCols,
        bCols
    );
   

    std::cout << "GPU-TIME MATMUL      NAIVE:  " << NaiveRes << std::endl;
    std::cout << "GPU-Time MatMul      TILED:  " << TiledRes << std::endl;

  
    return 0;
}
