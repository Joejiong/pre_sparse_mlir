#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <mkl.h>

#include <chrono>

#include "spdm.h"

#define M 1024
#define N 1024
#define K 1024
#define N_BLKSIZE 16
#define K_BLKSIZE 1
#define N_SPARSE 10
#define FLUSH_CACHE false

float A[M][K] __attribute__ ((aligned (64)));
float B[K][N] __attribute__ ((aligned (64)));
float C[M][N] __attribute__ ((aligned (64)));

BSCMatrix<float>* B_bsc_packed;
BSRMatrix<float>* B_bsr_packed;

void mkl_matmul() {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0, (float*)A, K, (float*)B, N, 0.0, (float*)C, N
    );
}

void init_input() {
    srand(0);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            A[m][k] = rand() % 5 - 2;
        }
    }   
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            B[k][n] = rand() % 11 - 5;
        }
    }
    // sparsify B
    for (int nb = 0; nb < N / N_BLKSIZE; nb++) {
        for (int kb = 0; kb < K / K_BLKSIZE; kb++) {
            bool zero_fill = rand() % N_SPARSE != 0;
            if (zero_fill) {
                for (int n = 0; n < N_BLKSIZE; n++) {
                    for (int k = 0; k < K_BLKSIZE; k++) {
                        B[kb*K_BLKSIZE + k][nb*N_BLKSIZE + n] = 0;
                    }
                }
            }
        }
    }
}

void spd_pack_B() {
    int shape[2] = {K,N};
    int blocksize[2] = {K_BLKSIZE,N_BLKSIZE};
    B_bsc_packed = create_bsc_matrix((float*)B, shape, blocksize);
    printf(
        "BSC: nnz: %d, total blocks: %d, sparsity: %f\n",
        B_bsc_packed->nnz, K*N/K_BLKSIZE/N_BLKSIZE, 1 - (float)B_bsc_packed->nnz/(K*N/K_BLKSIZE/N_BLKSIZE)
        );
    B_bsr_packed = create_bsr_matrix((float*)B, shape, blocksize);
    printf(
        "BSR: nnz: %d, total blocks: %d, sparsity: %f\n",
        B_bsr_packed->nnz, K*N/K_BLKSIZE/N_BLKSIZE, 1 - (float)B_bsr_packed->nnz/(K*N/K_BLKSIZE/N_BLKSIZE)
        );
}

void llc_flush() {
    static std::vector<char> llc;
    llc.resize(128*1024*1024);
    volatile char* data = llc.data();
    for (unsigned int i = 0; i < llc.size(); i++) {
      data[i]++;
    }
}

void spd_matmul_bsc() {
    assert(B_bsc_packed);
    spdm_16x1(M, N, K, (float*)A, B_bsc_packed, (float*)C);
}

void spd_matmul_bsr() {
    assert(B_bsr_packed);
    spdm(M, N, K, (float*)A, B_bsr_packed, (float*)C);
}

double bench_spdm_bsc(bool flush=true) {
    if (flush) llc_flush();
    auto begin = std::chrono::high_resolution_clock::now();
    spd_matmul_bsc();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

double bench_spdm_bsr(bool flush=true) {
    if (flush) llc_flush();
    auto begin = std::chrono::high_resolution_clock::now();
    spd_matmul_bsr();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

double bench_mkl(bool flush=true) {
    if (flush) llc_flush();
    auto begin = std::chrono::high_resolution_clock::now();
    mkl_matmul();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

float C_result[M][N];
float C_ref[M][N];

bool is_correct() {
    bool result_matched = true;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (C_result[m][n] != C_ref[m][n]) {
                printf("No match at (%d,%d): %f vs. %f (expected)\n", m, n, C_result[m][n], C_ref[m][n]);
                result_matched = false;
            }
        }
    }
    if (result_matched) {
        printf("Result matched!\n");
        return true;
    } else {
        return false;
    }
}

int main() {
    printf("M = %d, N = %d, K = %d, sparsity = %f, flush_cache = %d\n", M, N, K, 1-1.0 / N_SPARSE, FLUSH_CACHE);
    init_input();

    // correctness check first
    mkl_matmul();
    memcpy(C_ref, C, M*N*sizeof(float));
    spd_pack_B();
    spd_matmul_bsc();
    memcpy(C_result, C, M*N*sizeof(float));
    if (!is_correct()) {
        return 1;
    }
    spd_matmul_bsr();
    memcpy(C_result, C, M*N*sizeof(float));
    if (!is_correct()) {
        return 1;
    }

    // benchmarking
    const int num_tests = 1000;
    double totl = 0.0;
    double ms;
    for (int i = 0; i < num_tests; i++) {
        totl += bench_spdm_bsc(FLUSH_CACHE);
    }
    ms = totl / 1e6 / num_tests;
    printf("SPDM BSC time: %lfms, GFLOPS: %lf\n", ms, double(M)*N*K*2 / ms / 1e6);
#if 0
    totl = 0.0;
    for (int i = 0; i < num_tests; i++) {
        totl += bench_spdm_bsr(FLUSH_CACHE);
    }
    ms = totl / 1e6 / num_tests;
    printf("SPDM BSR time: %lfms\n", ms, double(M)*N*K*2 / ms / 1e6);
#endif
    totl = 0.0;
    for (int i = 0; i < num_tests; i++) {
        totl += bench_mkl(FLUSH_CACHE);
    }
    ms = totl / 1e6 / num_tests;
    printf("MKL time: %lfms, GFLOPS: %lf\n", ms, double(M)*N*K*2 / ms / 1e6);
    return 0;
}