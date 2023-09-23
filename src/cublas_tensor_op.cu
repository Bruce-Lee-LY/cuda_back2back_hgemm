// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: cublas tensor op back2back hgemm

#include "common.h"
#include "matrix.h"

cublasHandle_t getCublasTensorOpHandle() {
    cublasHandle_t handle = nullptr;
    HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    return handle;
}

void cublasTensorOp(half *A, half *B, half *C, half *D, size_t M, size_t L, size_t N, size_t K) {
    static cublasHandle_t handle = getCublasTensorOpHandle();
    static half alpha = 1.0;
    static half beta = 0.0;

    static Matrix *P = new Matrix(M, N, "Matrix P");

    HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                          CUDA_R_16F, K, &beta, P->getDevPtr(), CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, L, M, N, &alpha, C, CUDA_R_16F, N,
                                          P->getDevPtr(), CUDA_R_16F, N, &beta, D, CUDA_R_16F, L, CUBLAS_COMPUTE_16F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
