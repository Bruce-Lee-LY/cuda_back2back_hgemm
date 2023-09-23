// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: tester

#pragma once

#include "cuda_timer.h"
#include "matrix.h"

class Tester {
public:
    explicit Tester(size_t M = 512, size_t L = 2048, size_t N = 16, size_t K = 1024, size_t warmup_iterations = 1,
                    size_t profiling_iterations = 10, size_t sleep_duration = 100, bool enable_check = false)
        : m_M(M),
          m_L(L),
          m_N(N),
          m_K(K),
          m_warmup_iterations(warmup_iterations),
          m_profiling_iterations(profiling_iterations),
          m_sleep_duration(sleep_duration),
          m_enable_check(enable_check) {
        HGEMM_CHECK_GT(m_M, 0);
        HGEMM_CHECK_GT(m_L, 0);
        HGEMM_CHECK_GT(m_N, 0);
        HGEMM_CHECK_GT(m_K, 0);
        HGEMM_CHECK_GT(m_warmup_iterations, 0);
        HGEMM_CHECK_GT(m_profiling_iterations, 0);
        HGEMM_CHECK_GT(m_sleep_duration, 0);

        m_A = new Matrix(m_M, m_K, "Matrix A");
        HGEMM_CHECK(m_A);
        m_B = new Matrix(m_K, m_N, "Matrix B");
        HGEMM_CHECK(m_B);
        m_C = new Matrix(m_N, m_L, "Matrix C");
        HGEMM_CHECK(m_C);
        m_D = new Matrix(m_M, m_L, "Matrix D");
        HGEMM_CHECK(m_D);
        m_base = new Matrix(m_M, m_L, "Matrix Base");
        HGEMM_CHECK(m_base);

        if (m_enable_check) {
            m_cuda_timer.start();
            cublas_tensor_op(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_base->getDevPtr(), m_M, m_L, m_N,
                             m_K);
            HLOG("Cublas-Tensor-Op use: %.3f ms", m_cuda_timer.end());
            m_base->moveToHost();
            m_base->memSetDevice();
        }
    }

    ~Tester() {
        if (m_A) {
            delete m_A;
            m_A = nullptr;
        }

        if (m_B) {
            delete m_B;
            m_B = nullptr;
        }

        if (m_C) {
            delete m_C;
            m_C = nullptr;
        }

        if (m_D) {
            delete m_D;
            m_D = nullptr;
        }

        if (m_base) {
            delete m_base;
            m_base = nullptr;
        }
    }

    template <typename Func>
    void evaluate(Func &&b2b_hgemm, const std::string &name) {
        HLOG("----------------- Evaluating %s -----------------", name.c_str());
        usleep(m_sleep_duration * 1000);
        m_D->tearUp(m_base);

        // warm up
        m_cuda_timer.start();
        for (size_t i = 0; i < m_warmup_iterations; ++i) {
            b2b_hgemm(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_D->getDevPtr(), m_M, m_L, m_N, m_K);
        }
        m_warmup_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_warmup_iterations);
        HLOG("Warm up time: %.3f ms", m_warmup_time);

        if (m_enable_check) {
            m_D->moveToHost();
            m_D->checkValue(m_base);
        }

        profile(std::forward<Func>(b2b_hgemm), name);
    }

private:
    void cublas_tensor_op(half *A, half *B, half *C, half *D, size_t M, size_t L, size_t N, size_t K) {
        cublasHandle_t handle = nullptr;
        HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
        HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

        half alpha = 1.0;
        half beta = 0.0;

        Matrix *P = new Matrix(m_M, m_N, "Matrix P");
        HGEMM_CHECK(P);

        HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, K, A,
                                              CUDA_R_16F, K, &beta, P->getDevPtr(), CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, L, M, N, &alpha, C, CUDA_R_16F, N,
                                              P->getDevPtr(), CUDA_R_16F, N, &beta, D, CUDA_R_16F, L,
                                              CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (P) {
            delete P;
            P = nullptr;
        }
    }

    template <typename Func>
    void profile(Func &&b2b_hgemm, const std::string &name) {
        m_cuda_timer.start();
        for (size_t i = 0; i < m_profiling_iterations; ++i) {
            b2b_hgemm(m_A->getDevPtr(), m_B->getDevPtr(), m_C->getDevPtr(), m_D->getDevPtr(), m_M, m_L, m_N, m_K);
        }
        m_profiling_time = static_cast<double>(m_cuda_timer.end()) / static_cast<double>(m_profiling_iterations);
        m_throughput =
            static_cast<double>(m_M * m_N * m_K * 2) * 1e-12 / (static_cast<double>(m_profiling_time) * 1e-3);

        if ((std::abs(m_base_time) <= 1e-6) && (std::abs(m_base_throughput) <= 1e-6)) {
            m_base_time = m_profiling_time;
            m_base_throughput = m_throughput;
        }

        HLOG("%s exit, profiling time: %.3f ms (%.2f%%), throughput: %.3f TFLOPS (%.2f%%)", name.c_str(),
             m_profiling_time, m_profiling_time / m_base_time * 100, m_throughput,
             m_throughput / m_base_throughput * 100);
    }

    const size_t m_M = 512;
    const size_t m_L = 2048;
    const size_t m_N = 16;
    const size_t m_K = 1024;
    const size_t m_warmup_iterations = 1;
    const size_t m_profiling_iterations = 10;
    const size_t m_sleep_duration = 100;
    const bool m_enable_check = false;

    Matrix *m_A = nullptr;     // row major, M * K
    Matrix *m_B = nullptr;     // col major, K * N
    Matrix *m_C = nullptr;     // col major, N * L
    Matrix *m_D = nullptr;     // row major, M * L
    Matrix *m_base = nullptr;  // row major, M * L, base result, init matrix D before each back2back hgemm

    CudaTimer m_cuda_timer;

    double m_warmup_time = 0.0;
    double m_profiling_time = 0.0;
    double m_throughput = 0.0;
    double m_base_time = 0.0;        // cublas tensor op default
    double m_base_throughput = 0.0;  // cublas tensor op default

    HGEMM_DISALLOW_COPY_AND_ASSIGN(Tester);
};
