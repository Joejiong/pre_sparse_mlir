#pragma once

#include <string.h>
#include <assert.h>
#include <vector>
#include <cstdlib>
#include <immintrin.h>

// BSR definition follows scipy
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
// "data" is in (nnz, blocksize[0], blocksize[1])
// shape[0] -> blocksize[0], shape[1] -> blocksize[1]
// blocksize evenly divide corresponding shape
template <typename T>
struct BSRMatrix {
    int shape[2];
    int blocksize[2];
    int nnz;
    int nrowptr;
    T* data;
    int* colidxs;
    int* rowptr;
    int* nnzidxs;
};

template <typename T>
struct BSCMatrix {
    int shape[2];
    int blocksize[2];
    int nnz;
    int ncolptr;
    T* data;
    int* rowidxs;
    int* colptr;
    int* nnzidxs;
};

// The shape of dense_matrix is given by `shape[2]`.
// Sparse block granularity is given by `blocksize[2]`.
template <typename T>
BSRMatrix<T>* create_bsr_matrix(
    const T* dense_matrix, const int shape[2], const int blocksize[2])
{
    const int blksize = blocksize[0]*blocksize[1];
    BSRMatrix<T>* bsr_matrix = new BSRMatrix<T>;
    bsr_matrix->shape[0] = shape[0];
    bsr_matrix->shape[1] = shape[1];
    bsr_matrix->blocksize[0] = blocksize[0];
    bsr_matrix->blocksize[1] = blocksize[1];
    assert(shape[0] % blocksize[0] == 0);
    assert(shape[1] % blocksize[1] == 0);
    // Initialize rowptr and colidxs arrays
    // Dynamic arrays that will be copied to bsr_matrix after initialization
    std::vector<int> rowptr;
    std::vector<int> colidxs;
    for (int b_row = 0; b_row < bsr_matrix->shape[0] / blocksize[0]; b_row++) {
        rowptr.push_back(colidxs.size());
        for (int b_col = 0; b_col < bsr_matrix->shape[1] / blocksize[1]; b_col++) {
            // TODO: check zero
            bool is_zero = true;
            const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
            for (int i = 0; i < bsr_matrix->blocksize[0]; i++) {
                for (int j = 0; j < bsr_matrix->blocksize[1]; j++) {
                    if (dense_start[i * shape[1] + j] != 0) {
                        is_zero = false;
                        goto done_check_zero;
                    }
                }
            }
done_check_zero:
            if (!is_zero) {
                colidxs.push_back(b_col);
            }
        }
    }
    rowptr.push_back(colidxs.size());
    // init bsr_matrix->rowptr array
    bsr_matrix->nrowptr = rowptr.size();
    bsr_matrix->rowptr = new int[rowptr.size()];
    for (int i = 0; i < bsr_matrix->nrowptr; i++) {
        bsr_matrix->rowptr[i] = rowptr[i];
    }
    // init bsr_matrix->colidxs array
    bsr_matrix->nnz = colidxs.size();
    bsr_matrix->colidxs = new int[colidxs.size()];
    for (int i = 0; i < bsr_matrix->nnz; i++) {
        bsr_matrix->colidxs[i] = colidxs[i];
    }
    bsr_matrix->nnzidxs = new int[rowptr.size()-1];
    int nnzidx = 0;
    for (int i = 0; i < bsr_matrix->nrowptr-1; i++) {
        bsr_matrix->nnzidxs[i] = nnzidx;
        nnzidx += bsr_matrix->rowptr[i+1] - bsr_matrix->rowptr[i];
    }
    // init data matrix
    int nnz_idx = 0;
    bsr_matrix->data = (T*)aligned_alloc(64, bsr_matrix->nnz * blksize * sizeof(T));
    for (int b_row = 0; b_row < bsr_matrix->nrowptr-1; b_row++) {
        for (int b_col_idx = bsr_matrix->rowptr[b_row]; b_col_idx < bsr_matrix->rowptr[b_row+1]; b_col_idx++, nnz_idx++) {
            int b_col = bsr_matrix->colidxs[b_col_idx];
            T* blkstart = bsr_matrix->data + nnz_idx*blksize;
            const T* dense_start = dense_matrix + b_row * blocksize[0] * shape[1] + b_col * blocksize[1];
            for (int i = 0; i < bsr_matrix->blocksize[0]; i++) {
                for (int j = 0; j < bsr_matrix->blocksize[1]; j++) {
                    blkstart[i * bsr_matrix->blocksize[1] + j] = dense_start[i * shape[1] + j];
                }
            }
        }
    }
    return bsr_matrix;
}

template <typename T>
BSCMatrix<T>* create_bsc_matrix(
    const T* dense_matrix, const int shape[2], const int blocksize[2])
{
    BSRMatrix<T>* bsr = create_bsr_matrix(dense_matrix, shape, blocksize);
    BSCMatrix<T>* bsc = new BSCMatrix<T>;
    const int bs = blocksize[0] * blocksize[1];
    bsc->shape[0] = bsr->shape[0];
    bsc->shape[1] = bsr->shape[1];
    bsc->blocksize[0] = bsr->blocksize[0];
    bsc->blocksize[1] = bsr->blocksize[1];
    bsc->nnz = bsr->nnz;
    bsc->ncolptr = bsr->shape[1] / bsr->blocksize[1] + 1;
    bsc->data = (T*) aligned_alloc(64, bsr->nnz * bs * sizeof(T));
    bsc->colptr = new int[bsc->ncolptr];
    bsc->rowidxs = new int[bsr->nnz];
    // TODO: naive O(n^3) transpose, optimize it.
    int b_col = 0, ptr = 0;
    for (; b_col < bsc->ncolptr-1; b_col++) {
        bsc->colptr[b_col] = ptr;
        for (int b_row = 0, nnz_idx = 0; b_row < bsr->nrowptr-1; b_row++) {
            for (int b_col_idx = bsr->rowptr[b_row]; b_col_idx < bsr->rowptr[b_row+1]; b_col_idx++, nnz_idx++) {
                if (b_col == bsr->colidxs[b_col_idx]) {
                    memcpy(bsc->data + ptr * bs, bsr->data + nnz_idx * bs, sizeof(T) * bs);
                    bsc->rowidxs[ptr++] = b_row;
                }
            }
        }
    }
    bsc->colptr[b_col] = ptr;
    bsc->nnzidxs = new int[bsc->ncolptr-1];
    int nnzidx = 0;
    for (int i = 0; i < bsc->ncolptr-1; i++) {
        bsc->nnzidxs[i] = nnzidx;
        nnzidx += bsc->colptr[i+1] - bsc->colptr[i];
    }
    destroy_bsr_matrix(bsr);
    return bsc;
}

template <typename T>
void destroy_bsr_matrix(BSRMatrix<T>* bsr_matrix) {
    free(bsr_matrix->data);
    delete[] bsr_matrix->colidxs;
    delete[] bsr_matrix->rowptr;
    delete[] bsr_matrix->nnzidxs;
    delete bsr_matrix;
}

template <typename T>
void destroy_bsc_matrix(BSCMatrix<T>* bsc_matrix) {
    free(bsc_matrix->data);
    delete[] bsc_matrix->rowidxs;
    delete[] bsc_matrix->colptr;
    delete[] bsc_matrix->nnzidxs;
    delete bsc_matrix;
}

// Sparse and dense matrix multiplication
// TODO: support transposition
// TODO: support leading dimension
// TODO: support alpha and beta
// C = A * B, where A is in M*K, C is in M*N
// B is in BSR format with the shape (K, N)
template <typename T>
void spdm(
    int M, int N, int K,
    const T* A,
    const BSRMatrix<T>* B,
    T* C
) {
    const int m_bs = 16;
    const int simd_len = 16;
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);

    int K_blksize = B->blocksize[0];
    int N_blksize = B->blocksize[1];
    int blksize = N_blksize * K_blksize;
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        memset(C + m*N, 0, sizeof(T) * N);
        for (int b_row = 0, nnz_idx = 0; b_row < B->nrowptr-1; b_row++) { // K dim
            for (int b_col_idx = B->rowptr[b_row]; b_col_idx < B->rowptr[b_row+1]; b_col_idx++, nnz_idx++) { // N dim
                int b_col = B->colidxs[b_col_idx];
                for (int k = 0; k < K_blksize; k++) {
                    for (int n = 0; n < N_blksize / simd_len; n++) {
                        for (int ni = 0; ni < simd_len; ni++) {
                            C[m*N+b_col*N_blksize+n*simd_len+ni] += A[m*K+b_row*K_blksize+k] * B->data[nnz_idx*blksize+k*N_blksize+n*simd_len+ni];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void spdm(
    int M, int N, int K,
    const T* A,
    const BSCMatrix<T>* B,
    T* C
) {
    const int m_bs = 16;
    const int simd_len = 16;
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);

    int K_blksize = B->blocksize[0];
    int N_blksize = B->blocksize[1];
    int blksize = N_blksize * K_blksize;
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int b_col = 0, nnz_idx = 0; b_col < B->ncolptr-1; b_col++) { // N dim
            memset(C + m*N + b_col*16, 0, sizeof(T) * 16);
            for (int b_row_idx = B->colptr[b_col]; b_row_idx < B->colptr[b_col+1]; b_row_idx++, nnz_idx++) { // K dim
                int b_row = B->rowidxs[b_row_idx];
                for (int k = 0; k < K_blksize; k++) {
                    for (int n = 0; n < N_blksize / simd_len; n++) {
                        for (int ni = 0; ni < simd_len; ni++) {
                            C[m*N+b_col*N_blksize+n*simd_len+ni] += A[m*K+b_row*K_blksize+k] * B->data[nnz_idx*blksize+k*N_blksize+n*simd_len+ni];
                        }
                    }
                }
            }
        }
    }
}

#define SPDM_ALG 0

#if (SPDM_ALG == 0) // m blocking

template <typename T>
void spdm_16x1(
    int M, int N, int K,
    const T* A,
    const BSCMatrix<T>* B,
    T* C
) {
#define M_NBLK 4
    assert(B->blocksize[0] == 1);
    assert(B->blocksize[1] == 16);
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);
    assert(M % M_NBLK == 0);

    #pragma omp parallel for collapse(2)
    for (int mb = 0; mb < M / M_NBLK; mb++)
        for (int b_col = 0; b_col < B->ncolptr-1; b_col++) { // N dim
            __m512 c[M_NBLK];
            for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_setzero_ps();
            for (int b_row_idx = B->colptr[b_col], nnz_idx = B->nnzidxs[b_col]; b_row_idx < B->colptr[b_col+1]; b_row_idx++, nnz_idx++) { // K dim
                int b_row = B->rowidxs[b_row_idx];
                __m512 a[M_NBLK];
                for (int i = 0; i < M_NBLK; i++) a[i] = _mm512_set1_ps(A[(mb*M_NBLK+i)*K+b_row]);
                __m512 b = _mm512_load_ps(&B->data[nnz_idx*16]);
                for (int i = 0; i < M_NBLK; i++) c[i] = _mm512_fmadd_ps(b, a[i], c[i]);
            }
            for (int i = 0; i < M_NBLK; i++) _mm512_store_ps(C + (mb*M_NBLK+i)*N + b_col*16, c[i]);
        }
}

#else // no blocking

template <typename T>
void spdm_16x1(
    int M, int N, int K,
    const T* A,
    const BSCMatrix<T>* B,
    T* C
) {
    assert(B->blocksize[0] == 1);
    assert(B->blocksize[1] == 16);
    assert(K == B->shape[0]);
    assert(N == B->shape[1]);
    assert(K % B->blocksize[0] == 0);
    assert(N % B->blocksize[1] == 0);

    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++)
        for (int b_col = 0; b_col < B->ncolptr-1; b_col++) { // N dim
            __m512 c = _mm512_setzero_ps();
            for (int b_row_idx = B->colptr[b_col], nnz_idx = B->nnzidxs[b_col]; b_row_idx < B->colptr[b_col+1]; b_row_idx++, nnz_idx++) { // K dim
                int b_row = B->rowidxs[b_row_idx];
                __m512 a = _mm512_set1_ps(A[m*K+b_row]);
                __m512 b = _mm512_load_ps(&B->data[nnz_idx*16]);
                c = _mm512_fmadd_ps(b, a, c);
            }
            _mm512_store_ps(C + m*N + b_col*16, c);
        }
}

#endif
