#include "cuda_utils.h"
#include "matrix_mul.h"
#include "matrix_add.h"

#include <vector>
#include <iostream>
int main() {

    size_t numRows{ 100 };
    size_t numCols{ 100 };



    std::vector<int> A = generateMatrix(numRows, numCols, 1);
    std::vector<int> B = generateMatrix(numRows, numCols, 1);


    //For Add
    int cSize = numRows * numCols;
    int* C = new int[cSize];

    CudaAdd(A.data(), B.data(), C, A.size());

    
    if (!validateAdd(A.data(), B.data(), C, numRows * numCols))
    {
        std::cerr << "ADD KERNEL FAILED";
    }

    delete[] C;
    return 0;

}
