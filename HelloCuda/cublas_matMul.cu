#include "cublas_matMul.h"

float MatMulCUblas(CudaMatMulHandle& context)
{
	const float alpha = 1.0f;
	const float beta = 0.0f;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaEventRecord(start);

	//Since CUBLAS is col-major we need to transpose A and B
    cublasStatus_t status = cublasSgemm(
        context.cublasHandle,
        CUBLAS_OP_N,        // B not transposed
        CUBLAS_OP_N,        // A not transposed
        context.bCols,     // m = columns of C
        context.aRows,     // n = rows of C
        context.inner,     // k
        &alpha,
        context.dev_b,     // B first
        context.bCols,     // ldb
        context.dev_a,     // A second
        context.inner,     // lda
        &beta,
        context.dev_c,     // C
        context.bCols      // ldc
    );


    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS SGEMM failed\n";
        return -1.0f;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}