#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <mkl.h>

#include "microkernel.hpp"

#define M 1024
#define N 1024
#define K 1024
#define M_BLKSZ 32
#define N_BLKSZ 32
#define K_BLKSZ 32
#define M_NBLKS (M/M_BLKSZ)
#define N_NBLKS (N/N_BLKSZ)
#define K_NBLKS (K/K_BLKSZ)

#define OFFSET_2D(T, stride, x, y) ((float*)(T) + (x) * (stride) + (y))

float A[M][K];
float B[K][N];
float C[M][N];

float A_BLOCKED[M_NBLKS][K_NBLKS][M_BLKSZ][K_BLKSZ];
float B_BLOCKED[N_NBLKS][K_NBLKS][K_BLKSZ][N_BLKSZ];
float C_BLOCKED[M_NBLKS][N_NBLKS][M_BLKSZ][N_BLKSZ];

void ref_matmul() {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 1.0, (float*)A, K, (float*)B, N, 0.0, (float*)C, N
    );
}

void plain_matmul() {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M_NBLKS; m++) {
        for (int n = 0; n < N_NBLKS; n++) {
            dnnl_brgemm_init_f32(
                OFFSET_2D(C, N, m*M_BLKSZ, n*N_BLKSZ), M_BLKSZ, N_BLKSZ, N);
            for (int k = 0; k < K_NBLKS; k++) {
                dnnl_brgemm_update_f32(
                    OFFSET_2D(A, K, m*M_BLKSZ, k*K_BLKSZ),
                    OFFSET_2D(B, N, k*K_BLKSZ, n*N_BLKSZ),
                    OFFSET_2D(C, N, m*M_BLKSZ, n*N_BLKSZ),
                    /*num=*/1, M_BLKSZ, N_BLKSZ, K_BLKSZ,
                    /*LDA=*/K, /*LDB=*/N, /*LDC=*/N,
                    /*stride_a=*/M_BLKSZ*K_BLKSZ, /*stride_b=*/N_BLKSZ*K_BLKSZ);
            }
        }
    }
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
}

float C_result[M][N];
float C_ref[M][N];
int main() {
    init_input();

    // correctness check first
    plain_matmul();
    memcpy(C_result, C, M*N*sizeof(float));
    ref_matmul();
    memcpy(C_ref, C, M*N*sizeof(float));
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
    } else {
        return 1;
    }
    return 0;
}