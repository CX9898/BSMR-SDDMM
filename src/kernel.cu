#include <cstdio>

#include <mma.h>

#include "kernel.cuh"
#include "wmmaSetting.hpp"

const int WARP_SIZE = 32;

using namespace nvcuda::wmma;

template<typename T>
__global__ void test(int n, T *a) {
    for (int i = 0; i < n; ++i) {
        printf("%f ", a[i]);
    }
}

template __global__ void test<float>(int n, float *a);
template __global__ void test<half>(int n, half *a);

__global__ void convertFp32ToFp16(const int n, const float *in, half *out) {
    int idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
//        printf("in : %f, out : %f ", in[idx], out[idx]);
    }
}

__global__ void comp_sddmm_gpu(const int M, const int N, const int K,
                               const half *matrixA, const half *matrixB,
                               const float *matrixS,
                               float *matrixP) {
    const int warpM = (int) (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (int) (blockDim.x * blockIdx.x + threadIdx.x);

    // Compute dense matrix multiplication using Tensor core

    const int pRowId = warpM * WMMA_M;
    const int pColId = warpN * WMMA_N;

    if (pRowId >= M || pColId >= N) {
        return;
    }

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldp = N;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRowId = pRowId;
        const int aColId = kIter;

        const int bRowId = kIter;
        const int bColId = pColId;

        // Bounds checking
        if (aRowId < M && aColId < K && bRowId < K && bColId < N) {
            const auto aOffsetPtr = matrixA + aRowId * lda + aColId;
            const auto bOffsetPtr = matrixB + bRowId * ldb + bColId;

            load_matrix_sync(aFrag, aOffsetPtr, lda);
            load_matrix_sync(bFrag, bOffsetPtr, ldb);

            mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

//    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
//        const int sIdx = pRowId * ldc + pColId + idx;
//
//        cFrag.x[idx] *= matrixS[sIdx];
//    }
//    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
//
//        printf("%f ", aFrag.x[idx]);
//    }

    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
    store_matrix_sync(pOffsetPtr, cFrag, ldp, mem_row_major);
}
