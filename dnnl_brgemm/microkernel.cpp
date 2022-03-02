/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <cstring>
#include <dnnl.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <unordered_map>
//#include <util/utils.hpp>

#ifdef SC_KERNEL_PROFILE
#include <atomic>
#include <chrono>
extern std::atomic<uint64_t> mkernel_init;
extern std::atomic<uint64_t> mkernel_exec;
#endif

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// https://github.com/boostorg/container_hash/blob/master/include/boost/container_hash/hash.hpp
template <typename T>
static inline void hash_combine(size_t &seed, const T &v) {
    seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

using namespace dnnl::impl::cpu::x64;
struct brg_arg {
    float alpha;
    float beta;
    int LDA;
    int LDB;
    int LDC;
    int M;
    int N;
    int K;
    int stride_a;
    int stride_b;
    size_t hash;
    brgemm_batch_kind_t brg_type;

    brg_arg(float alpha, float beta, int LDA, int LDB, int LDC, int M, int N,
            int K, int stride_a, int stride_b, brgemm_batch_kind_t brg_type)
        : alpha(alpha)
        , beta(beta)
        , LDA(LDA)
        , LDB(LDB)
        , LDC(LDC)
        , M(M)
        , N(N)
        , K(K)
        , stride_a(stride_a)
        , stride_b(stride_b)
        , brg_type(brg_type) {
        hash = get_hash();
    }

    bool operator==(const brg_arg &v) const {
        return alpha == v.alpha && beta == v.beta && LDA == v.LDA
                && LDB == v.LDB && LDC == v.LDC && M == v.M && N == v.N
                && K == v.K && stride_a == v.stride_a && stride_b == v.stride_b
                && brg_type == v.brg_type;
    }

private:
    size_t get_hash() const {
        size_t seed = 0;
        hash_combine(seed, alpha);
        hash_combine(seed, beta);
        hash_combine(seed, LDA);
        hash_combine(seed, LDB);
        hash_combine(seed, LDC);
        hash_combine(seed, M);
        hash_combine(seed, N);
        hash_combine(seed, K);
        hash_combine(seed, stride_a);
        hash_combine(seed, stride_b);
        hash_combine(seed, (unsigned)brg_type);

        return seed;
    }
};

namespace std {
template <>
struct hash<brg_arg> {
    std::size_t operator()(const brg_arg &k) const { return k.hash; }
};
} // namespace std

struct brg_desc_safe {
    brg_desc_safe() {}
    ~brg_desc_safe() {
        for (auto &v : brg_desc_vec_) {
            brgemm_kernel_destroy(v.second);
        }
    }

    brgemm_kernel_t *get(const brg_arg &arg) {
        auto found_kernel = brg_desc_vec_.find(arg);
        if (found_kernel == brg_desc_vec_.end()) { return nullptr; }
        return found_kernel->second;
    }

    void set(const brg_arg &arg, brgemm_kernel_t *desc) {
        brg_desc_vec_.insert(std::make_pair(arg, desc));
    }

    brgemm_kernel_t *getInstance(const float &alpha, const float &beta,
            const int &LDA, const int &LDB, const int &LDC, const int &M,
            const int &N, const int &K, const int &stride_a,
            const int &stride_b, brgemm_batch_kind_t brg_type) {
        brg_arg arg {alpha, beta, LDA, LDB, LDC, M, N, K, stride_a, stride_b,
                brg_type};
        brgemm_kernel_t *found_kernel = get(arg);
        if (!found_kernel) {
            brgemm_t desc;
            // CHECK status
            brgemm_strides_t stride_info = {stride_a, stride_b};
            auto status = brgemm_desc_init(&desc, isa_any, brg_type, dnnl_f32,
                    dnnl_f32, false, false, brgemm_row_major, alpha, beta, LDA,
                    LDB, LDC, M, N, K, &stride_info);
            assert(status == dnnl::impl::status::success);
            brgemm_kernel_create(&found_kernel, desc);
            set(arg, found_kernel);
            return found_kernel;
        }
        return found_kernel;
    }

    std::unordered_map<brg_arg, brgemm_kernel_t *> brg_desc_vec_;
};

static thread_local brg_desc_safe g_brg_desc_s;

extern "C" void *dnnl_brgemm_f32_func(int M, int N, int K, int LDA, int LDB,
        int LDC, int stride_a, int stride_b, float beta) {
    float alpha = 1.0;
    return g_brg_desc_s.getInstance(alpha, beta, LDA, LDB, LDC, M, N, K,
            stride_a * sizeof(float), stride_b * sizeof(float), brgemm_strd);
}

extern "C" void dnnl_brgemm_f32_call(brgemm_kernel_t *brg_desc, const float *A,
        const float *B, float *C, int num) {
    brgemm_kernel_execute(brg_desc, num, (void **)A, (void **)B, (void *)C);
}

extern "C" void *dnnl_brgemm_list_f32_func(
        int M, int N, int K, int LDA, int LDB, int LDC, float beta) {
    float alpha = 1.0;
    if (M <= 0) { return nullptr; }
    return g_brg_desc_s.getInstance(
            alpha, beta, LDA, LDB, LDC, M, N, K, 0, 0, brgemm_addr);
}

extern "C" void dnnl_brgemm_list_f32_call(brgemm_kernel_t *brg_desc,
        const float **A_list, const float **B_list, float *C, int len, int num,
        int stride_a, int stride_b) {
    const float *A_addr[num * len];
    const float *B_addr[num * len];

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < num; ++j) {
            A_addr[i * num + j] = const_cast<float *>(&A_list[i][j * stride_a]);
            B_addr[i * num + j] = const_cast<float *>(&B_list[i][j * stride_b]);
        }
    }
    brgemm_kernel_execute(
            brg_desc, len * num, (void **)A_addr, (void **)B_addr, (void *)C);
}

extern "C" int dnnl_brgemm_init_update_f32(const float *A, const float *B,
        float *C, int num, int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b) {
// Assume that matrices are in row-major layout
// A: lda * m
// B: ldb * k
// C: ldc * m
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 0.0;
    auto brg_desc = g_brg_desc_s.getInstance(alpha, beta, LDA, LDB, LDC, M, N,
            K, stride_a * sizeof(float), stride_b * sizeof(float), brgemm_strd);
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    brgemm_kernel_execute(brg_desc, num, (void **)A, (void **)B, (void *)C);
#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();
    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}

extern "C" int dnnl_brgemm_init_f32(float *C, int M, int N, int LDC) {
    if (LDC == N) {
        memset(C, 0, M * N * sizeof(float));
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * LDC + j] = 0.0;
            }
        }
    }
    return 0;
}

extern "C" int dnnl_brgemm_update_f32(const float *A, const float *B, float *C,
        int num, int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b) {
    // Assume that matrices are in row-major layout
    // A: lda * m
    // B: ldb * k
    // C: ldc * m
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 1.0;
    auto brg_desc = g_brg_desc_s.getInstance(alpha, beta, LDA, LDB, LDC, M, N,
            K, stride_a * sizeof(float), stride_b * sizeof(float), brgemm_strd);
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    brgemm_kernel_execute(brg_desc, num, (void **)A, (void **)B, (void *)C);
#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();

    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}

extern "C" int dnnl_brgemm_list_update_f32(const float **A_list,
        const float **B_list, float *C, int len, int num, int M, int N, int K,
        int LDA, int LDB, int LDC, int stride_a, int stride_b) {
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 1.0;

    const float *A_addr[num * len];
    const float *B_addr[num * len];

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < num; ++j) {
            A_addr[i * num + j] = const_cast<float *>(&A_list[i][j * stride_a]);
            B_addr[i * num + j] = const_cast<float *>(&B_list[i][j * stride_b]);
        }
    }

    auto brg_desc = g_brg_desc_s.getInstance(
            alpha, beta, LDA, LDB, LDC, M, N, K, 0, 0, brgemm_addr);
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    brgemm_kernel_execute(
            brg_desc, len * num, (void **)A_addr, (void **)B_addr, (void *)C);
#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();

    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}
