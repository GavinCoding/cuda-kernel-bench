//Converts 2d Vector into 1d array
#include "cuda_utils.h"
#include <iostream>     //for std::cerr;
#include <random>



std::vector<float> flatten(const std::vector< std::vector<float> >& tdv)
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


    std::vector<float> flat(rowCount * colCount);
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
std::vector<float> generateMatrix(size_t rows, size_t cols, int seed)
{
    std::vector<float>  Matrix(rows*cols);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 10);

    for (size_t i = 0; i < rows*cols; ++i)
    {
        
         Matrix[i] = (float)dist(rng);
    }
    return Matrix;
}
bool validateAdd(const float* inputA, const float* inputB, const float* result, size_t N)
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



bool validateMultiply(const float* inputA, const float* inputB, const float* inputC, size_t aRows, size_t inner, size_t bCols)
{
   //Slow multiply A and B and check if it is equal to C
    
    float sum = 0;
    //Rows of A
    for (int rows = 0; rows < aRows; ++rows)
    {
        //Cols of B
        for (int cols = 0; cols < bCols; ++cols)
        {
            //Inner number of nums within each row of A and Col of B 
            for (int N = 0; N < inner; ++N)
            {
                sum += inputA[rows * inner + N] * inputB[N * bCols + cols];
            }
            if ((int)inputC[rows * bCols + cols] != (int)sum)
                return false;
            sum = 0;

        }

    }
    return true;
}