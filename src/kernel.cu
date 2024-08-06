#include <mma.h>

#include "kernel.cuh"
#include "matrixSetting.hpp"

const int WARP_SIZE = 32;

using namespace nvcuda::wmma;

__global__ void convertFp32ToFp16(const int n, const float *in, half *out) {
    int idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void compSddmm(const int M, const int N, const int K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP) {
    const int warpM = (int) (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (int) (blockDim.x * blockIdx.x + threadIdx.x);

    // Compute dense matrix multiplication using Tensor core

    const int cRow = warpM * WMMA_M;
    const int cCol = warpN * WMMA_N;

    if (cRow >= M || cCol >= N) {
        return;
    }

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    const int ldp = N;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRow = cRow;
        const int aCol = kIter;

        const int bRow = kIter;
        const int bCol = cCol;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            const auto aOffsetPtr = matrixA + aRow * lda + aCol;
            const auto bOffsetPtr = matrixB + bRow * ldb + bCol;

            load_matrix_sync(aFrag, aOffsetPtr, lda);
            load_matrix_sync(bFrag, bOffsetPtr, ldb);

            mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        const int sIdx = cRow * ldc + cCol + idx;

        cFrag.x[idx] *= matrixS[sIdx];
    }

    const auto pOffsetPtr = matrixP + cRow * ldc + cCol;
    store_matrix_sync(pOffsetPtr, cFrag, ldp, mem_row_major);
}
