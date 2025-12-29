#include "cuda_utils.h"
#include "matrix_mul.h"
#include "matrix_add.h"

#include <vector>
#include <iostream>
int main() {
    std::vector< std::vector<int> > A =
    {
        {1,2,3},
        {4,5,6},
        {7,8,9},
        {10,11,12},
        {13,14,15}
    };

    std::vector< std::vector<int> > B =
    {
        {4,6,2,10,12},
        {1,4,5,4,2},
        {3,6,7,12,3}
    };

    std::vector< std::vector<int> > D =
    {
        {1,2,3},
        {4,5,6},
        {7,8,9},
        {10,11,12},
        {13,14,15}
    };

    std::vector<int> A_flat = flatten(A);
    std::vector<int> B_flat = flatten(B);
    std::vector<int> D_flat = flatten(D);

    ////For Mult
    //int cSize = A.size() * B[0].size();
    //int* C = new int[cSize]
    //MatrixMultCuda(A_flat.data(), B_flat.data(), C, A.size(), B[0].size(), A[0].size());

    //For Add
    int cSize = A_flat.size();
    int* C = new int[cSize];

    CudaAdd(A_flat.data(), D_flat.data(), C, A_flat.size());

    for (int i = 0; i < cSize; i++)
    {
        std::cout << C[i] << " ";
        if ((i + 1) % (A[0].size()) == 0)
            std::cout << std::endl;
    }
    delete[] C;
    return 0;

}
std::vector< std::vector<int> > multiply(const std::vector<std::vector<int>> A, const std::vector<std::vector<int>> B)
{
    //Imagine we 
    //For Results
    size_t rowA = A.size();
    size_t colB = B[0].size();

    size_t colA = A[0].size();
    size_t rowB = B.size();


    //For Result
    std::vector< std::vector<int> > C(rowA, std::vector<int>(colB, 0));


    if (colA == rowB)
    {
        for (int i = 0; i < rowA; i++)
        {
            for (int j = 0; j < colB; j++)
            {
                for (int k = 0; k < rowB; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

    }
    else
    {
        std::cerr << "Can't Only multiple Maturx of size (nCmR) * (mCoR) ";
    }

    for (int RowNum = 0; RowNum < rowA; RowNum++)
    {
        for (int ColNum = 0; ColNum < colB; ColNum++)
        {
            std::cout << C[RowNum][ColNum] << " ";
        }
        std::cout << "\n";
    }

    return C;
}