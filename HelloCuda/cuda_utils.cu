//Converts 2d Vector into 1d array
#include "cuda_utils.h"
#include <iostream>     //for std::cerr;
#include <random>



std::vector<int> flatten(const std::vector< std::vector<int> >& tdv)
{
    if (tdv.empty()) {
        std::cerr << "flatten(): input has no rows\n";
        return {};
    }

    if (tdv[0].empty()) {
        std::cerr << "flatten(): input has no columns\n";
        return {};
    }

    size_t rowCount = tdv.size(); //number of columns within 2d vector
    size_t colCount = tdv[0].size();


    //Rectangluar Check
    for (size_t i = 1; i < rowCount; ++i)
    {
        if (tdv[i].size() != colCount)
        {
            std::cerr << "Input matrix is not Rextabgular";
            return {};
        }
    }


    std::vector<int> flat(rowCount * colCount);
    /*  std::cout << sizeof(flattened);*/

    size_t idx = 0;
    for (size_t i = 0; i < rowCount; ++i)
    {
        for (size_t j = 0; j < colCount; ++j)
        {
            flat[idx++] = tdv[i][j];
        }
    }





    return flat;

}
std::vector<int> generateMatrix(size_t rows, size_t cols, int seed)
{
    std::vector<int>  Matrix(rows*cols);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 10);

    for (size_t i = 0; i < rows*cols; ++i)
    {
        
         Matrix[i] = dist(rng);
    }
    return Matrix;
}
bool validateAdd(const int* inputA, const int* inputB, const int* result, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        if (result[i] != (inputA[i] + inputB[i]))
        {
            std::cout << result[i] << " != " <<inputA[i] << " + "<< inputB[i] << std::endl;
            return false;
        }
            
    }
    return true;
}


//Need To change this to work with flattened inputs
bool validateMultiply(const std::vector<std::vector<int>> A, const std::vector<std::vector<int>> B)
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

    return true;
}