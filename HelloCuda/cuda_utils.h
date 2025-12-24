#pragma once
#include <vector>

//Converts a 2d vector into a row-major 1d Vector
std::vector<int> flatten(const std::vector< std::vector<int> >& tdv);