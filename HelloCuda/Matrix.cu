#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>

__global__ void matrixMultiplyNaive(const std::vector<std::vector<int>>* A, const std::vector<std::vector<int>>* B, std::vector<std::vector<int>>* C);
void multiply(const std::vector<std::vector<int>>* A, const std::vector<std::vector<int>>* B, std::vector<std::vector<int>>* C);


int main()
{
	//Matrix multiply chill
	std::vector< std::vector<int> > A = {
		{5,3},
		{2,9}
	};
	std::vector< std::vector<int> > B{
		{7,6},
		{10,11}
	};


	std::vector< std::vector<int> > C;

}




void multiply(const std::vector<std::vector<int>>* A, const std::vector<std::vector<int>>* B, std::vector<std::vector<int>>* C)
{
	std::cout << sizeof(A);
}