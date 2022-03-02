extern "C" {
int dnnl_brgemm_init_update_f32(const float *A, const float *B, float *C,
        int num, int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b);
int dnnl_brgemm_init_f32(float *C, int M, int N, int LDC);
int dnnl_brgemm_update_f32(const float *A, const float *B, float *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b);
int dnnl_brgemm_list_update_f32(const float **A_list, const float **B_list,
        float *C, int len, int num, int M, int N, int K, int LDA, int LDB,
        int LDC, int stride_a, int stride_b);
}