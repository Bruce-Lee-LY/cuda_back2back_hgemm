// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: naive back2back hgemm

#include "common.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void simtNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, const half *__restrict__ C,
                                half *__restrict__ D, size_t M, size_t L, size_t N, size_t K) {
    const size_t row = blockIdx.x;

    if (row >= M) {
        return;
    }

    extern __shared__ half P[];

    const size_t P_lane_valid_num = blockDim.x <= N ? blockDim.x : N;
    if (threadIdx.x < P_lane_valid_num) {
        const size_t P_lane_len = N / P_lane_valid_num;
        const size_t P_lane_start = threadIdx.x * P_lane_len;
        const size_t P_lane_end = (threadIdx.x + 1) * P_lane_len;

#pragma unroll
        for (size_t j = P_lane_start; j < P_lane_end; ++j) {
            half tmp0 = 0.0;
#pragma unroll
            for (size_t k = 0; k < K; ++k) {
                tmp0 += A[row * K + k] * B[k + j * K];
            }
            P[j] = tmp0;
        }
    }

    __syncthreads();

    const size_t D_lane_valid_num = blockDim.x <= L ? blockDim.x : L;
    if (threadIdx.x >= D_lane_valid_num) {
        return;
    }

    const size_t D_lane_len = L / D_lane_valid_num;
    const size_t D_lane_start = threadIdx.x * D_lane_len;
    const size_t D_lane_end = (threadIdx.x + 1) * D_lane_len;

#pragma unroll
    for (size_t jj = D_lane_start; jj < D_lane_end; ++jj) {
        half tmp1 = 0.0;
#pragma unroll
        for (size_t kk = 0; kk < N; ++kk) {
            tmp1 += P[kk] * C[kk + jj * N];
        }
        D[row * L + jj] = tmp1;
    }
}

size_t initSimtNaive(size_t N) {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = N * sizeof(half);
    HLOG("smem_max_size: %.0f KBytes (%zu bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(simtNaiveKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void simtNaive(half *A, half *B, half *C, half *D, size_t M, size_t L, size_t N, size_t K) {
    static size_t smem_max_size = initSimtNaive(N);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(M);

    simtNaiveKernel<<<grid, block, smem_max_size>>>(A, B, C, D, M, L, N, K);
}
