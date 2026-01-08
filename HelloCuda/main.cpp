#include "cuda_utils.h"
#include "matrix_mul.h"
#include "matrix_add.h"
#include <iomanip>

#include <vector>
#include <iostream>

int main() {

    constexpr size_t size = 4;  // square matrix (10x10)
    constexpr int tileSize = TILESIZE;


    const int aRows = 4;
    const int aCols = 6;
    const int bRows = 6;
    const int bCols = 7;

    bool isBig = false;

    if ((size_t)((size_t)(aRows * bCols)) * bRows > 1000000)
    {
        isBig = true;
    }

    if (aCols != bRows)
    {
        std::cerr << "Matrices A and B are not Compatible for multiplication.";
    }
    

    std::vector<float> A = generateMatrix(aRows, aCols, 1);
    std::vector<float> B = generateMatrix(bRows, bCols, 2);

    std::vector<float> C(aRows * bCols, 0);

    // Run tiled matrix multiplication
    cudaError_t result = MatrixMultCuda(
        A.data(),
        B.data(),
        C.data(),
        aRows,
        aCols,
        bCols
    );

    if (result != cudaSuccess) {
        std::cerr << "Matrix multiply kernel failed: "
            << cudaGetErrorString(result) << "\n";
        return 1;
    }


    //If it's not too big we can Print and check if it's valid
    if (!isBig)
    {
        //if it's small and not valid Don't print
        if (!validateMultiply(A.data(), B.data(), C.data(), aRows, aCols, bCols))
        {
            std::cout << "\nMATMUL RESULT INCORRECT\n";
        }
        else
        {
            // Optional: print a small section to verify visually
            std::cout << "A:\n";
            for (int i = 0; i < aRows; ++i) {
                for (int j = 0; j < aCols; ++j) {
                    std::cout << std::setw(3) << A[i * aCols + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n\nB:\n";
            for (int i = 0; i < bRows; ++i) {
                for (int j = 0; j < bCols; ++j) {
                    std::cout << std::setw(3) << B[i * bCols + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n\nC:\n";
            for (int i = 0; i < aRows; ++i) {
                for (int j = 0; j < bCols; ++j) {
                    std::cout << std::setw(4) << C[i * aRows + j] << " ";
                }
                std::cout << "\n";
            }
        }
    }
    
   

    return 0;
}
