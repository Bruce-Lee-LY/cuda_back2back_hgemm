// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 13:07:04 on Sun, Jul 30, 2023
//
// Description: mma async shm back2back hgemm b128x8_w16x8_p128x16_w16x16_k128

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define GEMM0_BLOCK_ROWS 128
#define GEMM0_BLOCK_COLS 16

#define GEMM0_WARP_ROWS 16
#define GEMM0_WARP_COLS 16

#define GEMM0_BLOCK_ROW_WARPS 1  // GEMM0_BLOCK_COLS / GEMM0_WARP_COLS
#define GEMM0_BLOCK_COL_WARPS 8  // GEMM0_BLOCK_ROWS / GEMM0_WARP_ROWS

#define GEMM0_BLOCK_ROW_TILES 2  // GEMM0_BLOCK_COLS / MMA_N
#define GEMM0_BLOCK_COL_TILES 8  // GEMM0_BLOCK_ROWS / MMA_M

#define GEMM0_WARP_ROW_TILES 2  // GEMM0_WARP_COLS / MMA_N
#define GEMM0_WARP_COL_TILES 1  // GEMM0_WARP_ROWS / MMA_M

#define GEMM1_BLOCK_ROWS 128
#define GEMM1_BLOCK_COLS 8

#define GEMM1_WARP_ROWS 16
#define GEMM1_WARP_COLS 8

#define GEMM1_BLOCK_ROW_WARPS 1  // GEMM1_BLOCK_COLS / GEMM1_WARP_COLS
#define GEMM1_BLOCK_COL_WARPS 8  // GEMM1_BLOCK_ROWS / GEMM1_WARP_ROWS

#define GEMM1_BLOCK_ROW_TILES 1  // GEMM1_BLOCK_COLS / MMA_N
#define GEMM1_BLOCK_COL_TILES 8  // GEMM1_BLOCK_ROWS / MMA_M

#define GEMM1_WARP_ROW_TILES 1  // GEMM1_WARP_COLS / MMA_N
#define GEMM1_WARP_COL_TILES 1  // GEMM1_WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // GEMM0_BLOCK_ROW_WARPS * GEMM0_BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define GEMM0_CHUNK_K 8  // 128 / MMA_K

#define THREAD_COPY_BYTES 16

#define GEMM0_CHUNK_LINE_BYTES 256         // GEMM0_CHUNK_K * MMA_K * sizeof(half)
#define GEMM0_CHUNK_COPY_LINES_PER_WARP 2  // (WARP_SIZE * THREAD_COPY_BYTES) / GEMM0_CHUNK_LINE_BYTES
#define GEMM0_CHUNK_COPY_LINE_LANES 16     // WARP_SIZE / GEMM0_CHUNK_COPY_LINES_PER_WARP

#define SMEM_PADDING 8

#define AB_SMEM_STRIDE 136  // GEMM0_CHUNK_K * MMA_K + SMEM_PADDING

#define PC_SMEM_STRIDE 24  // GEMM0_BLOCK_COLS + SMEM_PADDING

#define D_SMEM_STRIDE 16  // GEMM1_BLOCK_COLS + SMEM_PADDING
#define D_SMEM_OFFSET 8   // GEMM1_WARP_COLS

__global__ void mmaAsyncSmemKernel(const half *__restrict__ A, const half *__restrict__ B, const half *__restrict__ C,
                                   half *__restrict__ D, size_t M, size_t L, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t L_tiles = div_ceil(L, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i = blockIdx.y * GEMM1_BLOCK_COL_TILES;
    const size_t block_tile_j = blockIdx.x * GEMM1_BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= L_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];
    __shared__ half PC_smem[GEMM1_BLOCK_ROWS + GEMM1_BLOCK_COLS][PC_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = GEMM0_BLOCK_ROWS;
    constexpr size_t C_smem_idx_off = GEMM1_BLOCK_ROWS;

    const size_t P_smem_warp_tile_idx = warp_id * GEMM0_WARP_ROWS;

    half *smem_warp_tile_row_ptr = &smem[0][0] + (warp_id / GEMM1_BLOCK_ROW_WARPS) * D_SMEM_STRIDE * GEMM1_WARP_ROWS;

    half *smem_warp_stream_ptr = &smem[0][0] + warp_id * GEMM1_WARP_ROWS * D_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i * MMA_M + warp_id * GEMM1_WARP_ROWS) * L + block_tile_j * MMA_N;
    half *src_gmem_warp_stream_ptr = &D[gmem_idx];

    uint32_t RP0[GEMM0_WARP_COL_TILES][GEMM0_WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
            RP0[i][j][0] = 0;
            RP0[i][j][1] = 0;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + GEMM0_BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[0] + GEMM0_BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = GEMM0_BLOCK_ROWS / (GEMM0_CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = GEMM0_BLOCK_COLS / (GEMM0_CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += GEMM0_CHUNK_K) {
        size_t A_smem_idx = GEMM0_BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / GEMM0_CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % GEMM0_CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / GEMM0_CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        (lane_id % GEMM0_CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + GEMM0_CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += GEMM0_CHUNK_COPY_LINES_PER_WARP;
        }

        size_t B_smem_idx = B_smem_idx_off + GEMM0_BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / GEMM0_CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % GEMM0_CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / GEMM0_CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        (lane_id % GEMM0_CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + GEMM0_CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += GEMM0_CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < GEMM0_CHUNK_K; ++k_step) {
            uint32_t RA[GEMM0_WARP_COL_TILES][4];
            uint32_t RB[GEMM0_WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / GEMM0_BLOCK_ROW_WARPS) * GEMM0_WARP_ROWS + i * MMA_M;
                uint32_t A_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[A_smem_idx + lane_id % 16][k_step * MMA_K + (lane_id / 16) * 8]);

                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
            }

#pragma unroll
            for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
                size_t B_smem_idx = B_smem_idx_off + (warp_id % GEMM0_BLOCK_ROW_WARPS) * GEMM0_WARP_COLS + j * MMA_N;
                uint32_t B_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[B_smem_idx + lane_id % 8][k_step * MMA_K + ((lane_id / 8) % 2) * 8]);

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
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < GEMM0_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM0_WARP_ROW_TILES; ++j) {
            *((uint32_t *)(&PC_smem[P_smem_warp_tile_idx + i * MMA_M + lane_id / 4][j * MMA_N]) + lane_id % 4) =
                RP0[i][j][0];
            *((uint32_t *)(&PC_smem[P_smem_warp_tile_idx + i * MMA_M + lane_id / 4 + 8][j * MMA_N]) + lane_id % 4) =
                RP0[i][j][1];
        }
    }

    __syncthreads();

    uint32_t RD[GEMM1_WARP_COL_TILES][GEMM1_WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < GEMM1_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM1_WARP_ROW_TILES; ++j) {
            RD[i][j][0] = 0;
            RD[i][j][1] = 0;
        }
    }

    if (warp_id == 0 && lane_id < MMA_N * 2) {
        int4 *C_lane_ptr = (int4 *)(&C[(block_tile_j * MMA_N + lane_id / 2) * N]) + lane_id % 2;
        uint32_t C_smem_lane_addr =
            __cvta_generic_to_shared(&PC_smem[C_smem_idx_off + lane_id / 2][0]) + (lane_id % 2) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(C_smem_lane_addr, C_lane_ptr, THREAD_COPY_BYTES);
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();

    uint32_t RP1[GEMM1_WARP_COL_TILES][4];
    uint32_t RC[GEMM1_WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < GEMM1_WARP_COL_TILES; ++i) {
        size_t P_smem_idx = (warp_id / GEMM1_BLOCK_ROW_WARPS) * GEMM1_WARP_ROWS + i * MMA_M;
        uint32_t P_smem_lane_addr = __cvta_generic_to_shared(&PC_smem[P_smem_idx + lane_id % 16][(lane_id / 16) * 8]);

        LDMATRIX_X4(RP1[i][0], RP1[i][1], RP1[i][2], RP1[i][3], P_smem_lane_addr);
    }

#pragma unroll
    for (size_t j = 0; j < GEMM1_WARP_ROW_TILES; ++j) {
        size_t C_smem_idx = C_smem_idx_off + (warp_id % GEMM1_BLOCK_ROW_WARPS) * GEMM1_WARP_COLS + j * MMA_N;
        uint32_t C_smem_lane_addr =
            __cvta_generic_to_shared(&PC_smem[C_smem_idx + lane_id % 8][((lane_id / 8) % 2) * 8]);

        LDMATRIX_X2(RC[j][0], RC[j][1], C_smem_lane_addr);
    }

#pragma unroll
    for (size_t i = 0; i < GEMM1_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM1_WARP_ROW_TILES; ++j) {
            HMMA16816(RD[i][j][0], RD[i][j][1], RP1[i][0], RP1[i][1], RP1[i][2], RP1[i][3], RC[j][0], RC[j][1],
                      RD[i][j][0], RD[i][j][1]);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < GEMM1_WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < GEMM1_WARP_ROW_TILES; ++j) {
            half *lane_ptr0 = smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * D_SMEM_STRIDE +
                              (warp_id % GEMM1_BLOCK_ROW_WARPS) * D_SMEM_OFFSET + j * MMA_N +
                              (lane_id % 4) * sizeof(uint32_t) / sizeof(half);
            half *lane_ptr1 = smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * D_SMEM_STRIDE +
                              (warp_id % GEMM1_BLOCK_ROW_WARPS) * D_SMEM_OFFSET + j * MMA_N +
                              (lane_id % 4) * sizeof(uint32_t) / sizeof(half);

            *((uint32_t *)(lane_ptr0)) = RD[i][j][0];
            *((uint32_t *)(lane_ptr1)) = RD[i][j][1];
        }
    }

    __syncthreads();

    *((int2 *)(src_gmem_warp_stream_ptr + (lane_id / 2) * L) + lane_id % 2) =
        *((int2 *)(smem_warp_stream_ptr + (lane_id / 2) * D_SMEM_STRIDE) + lane_id % 2);
}

size_t initMmaAsyncSmem() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t dynamic_smem_size = std::max((GEMM0_BLOCK_ROWS + GEMM0_BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half),
                                        GEMM1_BLOCK_ROWS * D_SMEM_STRIDE * sizeof(half));
    size_t smem_max_size = dynamic_smem_size + (GEMM1_BLOCK_ROWS + GEMM1_BLOCK_COLS) * PC_SMEM_STRIDE * sizeof(half);
    HLOG("smem_max_size: %.0f KBytes (%zu bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaAsyncSmemKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaAsyncSmem(half *A, half *B, half *C, half *D, size_t M, size_t L, size_t N, size_t K) {
    static size_t smem_max_size = initMmaAsyncSmem();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(L, GEMM1_BLOCK_COLS), div_ceil(M, GEMM1_BLOCK_ROWS));

    mmaAsyncSmemKernel<<<grid, block, smem_max_size>>>(A, B, C, D, M, L, N, K);
}
