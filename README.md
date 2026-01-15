# CUDA Kernels

## Overview
Basic CUDA kernels written for the purpose of exploring optimization techniques and benchmarking. Thus far, this project is focused on matrix multiplication algorithms and kernel launch dimension optimizations.

## Motivation
A variety of kernels written to explore performance characteristics and benchmark against both CPU implementations and cuBLAS. The goal is learning and exploration—building a foundation of kernels that enables deeper investigation into optimization strategies, benchmarking methodology, and standardized performance analytics and reporting.

## Performance and Analytics
This project emphasizes quantitative performance analysis. Kernels are evaluated across multiple dimensions, including execution time, and the impact of launch configuration and memory access patterns. As optimizations are introduced, results are compared against prior implementations to highlight measurable performance gains. Times measured represent GPU time only. Seperation of context and memory management allow for strict timing using the cuda events system.
### Matrix generation
Matrices are generated using a Mersenne Twister pseudo-random number generator (std::mt19937) and initially represented as std::vector<std::vector<int>> on the host.
The data is then flattened into a contiguous one-dimensional buffer and copied to device memory for CUDA kernel execution.
Benchmarks use matrices larger than 10⁶ elements to ensure kernel execution dominates fixed overhead costs and to more accurately reflect computational efficiency.
## Implementation Highlights
- Incremental kernel development, starting from naive implementations and progressing toward optimized versions  
- Exploration of kernel launch dimensions and their impact on occupancy and throughput  


## Future Work
Planned extensions include additional optimization techniques, expanded benchmarking coverage, and more standardized reporting of performance metrics. Future kernels will further explore memory hierarchy utilization, scalability, and advanced CUDA optimization strategies. Comparison against highly optimized library implementations (cuBLAS) to contextualize performance results  


## Performance Results
All Benchmarks on ran on an NVIDIA RTX 4070 running CUDA 12.8.
Each test reports the minimum, average, and maximum runtime over 15 runs.
| TEST | A ROWS | INNER | B COLS | TILEWIDTH | BLOCK | NAIVE MIN | NAIVE AVG | NAIVE MAX | TILED MIN | TILED AVG | TILED MAX | AVG SPEEDUP |
| ---- | ------ | ----- | ------ | --------- | ----- | --------- | --------- | --------- | --------- | --------- | --------- | ----------- |
| 1    | 512    | 512   | 512    | 32        | 32x32 | 2.865     | 4.629     | 16.759    | 2.137     | 3.122     | 11.185    | **1.483x**      |
| 2    | 1024   | 1024  | 1024   | 32        | 32x32 | 21.34     | 23.706    | 28.752    | 15.271    | 17.16     | 21.723    | **1.381x**      |
| 3    | 2048   | 2048  | 2048   | 32        | 32x32 | 180.849   | 185.788   | 211.494   | 123.957   | 126.055   | 132.709   | **1.474x**      |
| 4    | 1024   | 1024  | 1024   | 16        | 16x16 | 15.95     | 17.608    | 21.104    | 17.284    | 19.097    | 23.574    | **0.922x**      |
| 5    | 2048   | 2048  | 2048   | 16        | 16x16 | 128.267   | 132.681   | 158.708   | 136.386   | 139.301   | 148.56    | **0.952x**      |


Next, I unrolled the tiled kernel to eliminate the overhead of the inner loop. This optimization yielded up to a 1.89× speedup over the looped tiled matrix multiplication for square matrices of size 2048. Additionally, I added support for cuBLAS SGEMM to benchmark my custom kernels against a highly optimized, production-grade implementation.

|       Type      | Best   | Avg    | Worst  | % of cuBLAS |
| --------------- | ------ | ------ | ------ | ----------- |
| Naive           | 183.78 | 190.02 | 245.26 | 3.3         |
| Tiled Unwrapped | 63.91  | 66.6   | 81.582 | 9.43        |
| cuBLAS SGEMM    | 0.88   | 6.28   | 78.698 | 100         |
