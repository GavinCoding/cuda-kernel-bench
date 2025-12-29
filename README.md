# CUDA Kernels

## Overview
Basic CUDA kernels written for the purpose of exploring optimization techniques and benchmarking. Thus far, this project is focused on matrix multiplication algorithms and kernel launch dimension optimizations.

## Motivation
A variety of kernels written to explore performance characteristics and benchmark against both CPU implementations and cuBLAS. The goal is learning and exploration—building a foundation of kernels that enables deeper investigation into optimization strategies, benchmarking methodology, and standardized performance analytics and reporting.

## Performance and Analytics
This project emphasizes quantitative performance analysis. Kernels are evaluated across multiple dimensions, including execution time, speedup relative to CPU baselines, and the impact of launch configuration and memory access patterns. As optimizations are introduced, results are compared against prior implementations to highlight measurable performance gains.
### Matrix generation
Matrices are generated using a Mersenne Twister pseudo-random number generator (std::mt19937) and initially represented as std::vector<std::vector<int>> on the host.
The data is then flattened into a contiguous one-dimensional buffer and copied to device memory for CUDA kernel execution.
Benchmarks use matrices larger than 10⁶ elements to ensure kernel execution dominates fixed overhead costs and to more accurately reflect computational efficiency.
## Implementation Highlights
- Incremental kernel development, starting from naive implementations and progressing toward optimized versions  
- Exploration of kernel launch dimensions and their impact on occupancy and throughput  
- Comparison against highly optimized library implementations (cuBLAS) to contextualize performance results  

## How to Run
Build and run instructions are provided for reproducing benchmarks and performance measurements. All results are generated using consistent benchmarking methodology to ensure fair comparison across implementations.

## Future Work
Planned extensions include additional optimization techniques, expanded benchmarking coverage, and more standardized reporting of performance metrics. Future kernels will further explore memory hierarchy utilization, scalability, and advanced CUDA optimization strategies.
