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
            return false;
    }
    return true;
}