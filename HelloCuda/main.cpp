#include "cuda_utils.h"
#include "matrix_mul.h"
#include "matrix_add.h"

#include <vector>
#include <iostream>

int main() {

    constexpr size_t size = 4;  // square matrix (10x10)
    constexpr int tileSize = TILESIZE;

    std::vector<int> A = generateMatrix(size, size, 1);
    std::vector<int> B = generateMatrix(size, size, 1);

    std::vector<int> C(size * size, 0);

    // Run tiled matrix multiplication
    cudaError_t result = MatrixMultCuda(
        A.data(),
        B.data(),
        C.data(),
        size,
        size,
        size
    );

    if (result != cudaSuccess) {
        std::cerr << "Matrix multiply kernel failed: "
            << cudaGetErrorString(result) << "\n";
        return 1;
    }

    // Optional: print a small section to verify visually
    std::cout << "A:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << A[i * size + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\nB:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << B[i * size + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\nC:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << C[i * size + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
