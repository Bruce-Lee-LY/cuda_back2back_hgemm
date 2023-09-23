// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 13:07:04 on Sun, Jul 30, 2023
//
// Description: mma naive shm back2back hgemm b16x8_w16x8_p16x16_w16x16

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define GEMM0_WARP_ROWS 16  // MMA_M
#define GEMM0_WARP_COLS 16  // MMA_K

#define GEMM0_WARP_ROW_TILES 2  // GEMM0_WARP_COLS / MMA_N
#define GEMM0_WARP_COL_TILES 1  // GEMM0_WARP_ROWS / MMA_M

#define GEMM1_WARP_ROWS 16  // MMA_M
#define GEMM1_WARP_COLS 8   // MMA_N

#define GEMM1_WARP_ROW_TILES 1  // GEMM1_WARP_COLS / MMA_N
#define GEMM1_WARP_COL_TILES 1  // GEMM1_WARP_ROWS / MMA_M

#define WARP_SIZE 32

#define GEMM0_CHUNK_K 16  // MMA_K

#define GEMM1_CHUNK_K 16  // MMA_K

__global__ void mmaNaiveSmemKernel(const half *__restrict__ A, const half *__restrict__ B, const half *__restrict__ C,
                                   half *__restrict__ D, size_t M, size_t L, size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= L) {
        return;
    }

    __shared__ half A_smem[GEMM0_WARP_ROWS][GEMM0_CHUNK_K];
    __shared__ half B_smem[GEMM0_WARP_COLS][GEMM0_CHUNK_K];
    __shared__ half P_smem[GEMM0_WARP_ROWS][GEMM0_WARP_COLS];

    __shared__ half C_smem[GEMM1_WARP_COLS][GEMM1_CHUNK_K];
    __shared__ half D_smem[GEMM1_WARP_ROWS][GEMM1_WARP_COLS];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RP0[GEMM0_WARP_COL_TILES][GEMM0_WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
            RP0[i][j][0] = 0;
            RP0[i][j][1] = 0;
        }
    }

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&B[i * MMA_K + (lane_id / 2) * K]) + lane_id % 2);

        __syncthreads();

        uint32_t RA[GEMM0_WARP_COL_TILES][4];
        uint32_t RB[GEMM0_WARP_ROW_TILES][2];

#pragma unroll
        for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[i * MMA_M + lane_id % 16][(lane_id / 16) * 8]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
            uint32_t B_smem_lane_addr =
                __cvta_generic_to_shared(&B_smem[j * MMA_N + lane_id % 8][((lane_id / 8) % 2) * 8]);
            LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
                HMMA16816(RP0[i][j][0], RP0[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                          RP0[i][j][0], RP0[i][j][1]);
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
            *((uint32_t *)(&P_smem[i * MMA_M + lane_id / 4][j * MMA_N]) + lane_id % 4) = RP0[i][j][0];
            *((uint32_t *)(&P_smem[i * MMA_M + lane_id / 4 + 8][j * MMA_N]) + lane_id % 4) = RP0[i][j][1];
        }
    }

    __syncthreads();

    if (lane_id < MMA_N * 2) {
        *((int4 *)(&C_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&C[(warp_col + lane_id / 2) * N]) + lane_id % 2);
    }

    __syncthreads();

    uint32_t RP1[4];
    uint32_t RC[2];
    uint32_t RD[2] = {0, 0};

    uint32_t P_smem_lane_addr = __cvta_generic_to_shared(&P_smem[lane_id % 16][(lane_id / 16) * 8]);
    LDMATRIX_X4(RP1[0], RP1[1], RP1[2], RP1[3], P_smem_lane_addr);

    uint32_t C_smem_lane_addr = __cvta_generic_to_shared(&C_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
    LDMATRIX_X2(RC[0], RC[1], C_smem_lane_addr);

    HMMA16816(RD[0], RD[1], RP1[0], RP1[1], RP1[2], RP1[3], RC[0], RC[1], RD[0], RD[1]);

    __syncthreads();

    *((uint32_t *)(&D_smem[lane_id / 4][0]) + lane_id % 4) = RD[0];
    *((uint32_t *)(&D_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RD[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&D[(warp_row + lane_id) * L + warp_col])) = *((int4 *)(&D_smem[lane_id][0]));
    }
}

void mmaNaiveSmem(half *A, half *B, half *C, half *D, size_t M, size_t L, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(L, GEMM1_WARP_COLS), div_ceil(M, GEMM1_WARP_ROWS));

    mmaNaiveSmemKernel<<<grid, block>>>(A, B, C, D, M, L, N, K);
}
