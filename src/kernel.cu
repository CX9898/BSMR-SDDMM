#include <cstdio>

#include <mma.h>

#include "kernel.cuh"
#include "wmmaSetting.hpp"
#include "Matrix.hpp"

const float SPARSITY_BOUND = 0.5f;

template<typename T>
__device__ int sparsityComparator(const int WMMA_M,
                                  const int WMMA_N,
                                  const int ld,
                                  const MatrixStorageOrder storageOrder,
                                  const T *matrixPtr) {
    int nnzCount = 0;
#pragma unroll
    for (int rowIter = 0; rowIter < WMMA_M; ++rowIter) {
#pragma unroll
        for (int colIter = 0; colIter < WMMA_N; ++colIter) {
            if (storageOrder == MatrixStorageOrder::row_major) {
                nnzCount += *(matrixPtr + rowIter * ld + colIter) == 0 ? 0 : 1;
            } else {
                nnzCount += *(matrixPtr + colIter * ld + rowIter) == 0 ? 0 : 1;
            }
        }
    }
    const int numValues = WMMA_M * WMMA_N;
    const float sparsity = static_cast<float>(numValues - nnzCount) / numValues;
    return sparsity >= SPARSITY_BOUND ? 1 : 0;
}

template<typename T>
__global__ void printData(int n, T *a) {
    for (int i = 0; i < n; ++i) {
        printf("%f ", static_cast<float>(a[i]));
    }
}

template __global__ void printData<float>(int n, float *a);
template __global__ void printData<half>(int n, half *a);

__global__ void convertFp32ToFp16(const int n, const float *in, half *out) {
    int idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void comp_sddmm_gpu(const int M, const int N, const int K,
                               const half *matrixA, const half *matrixB,
                               const float *matrixS,
                               float *matrixP) {
    const int tidX = (blockDim.x * blockIdx.x + threadIdx.x);
    const int tidY = (blockDim.y * blockIdx.y + threadIdx.y);

    const int warpM = (int) (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (int) (blockDim.y * blockIdx.y + threadIdx.y);

    // Compute dense matrix multiplication using Tensor core

    const int pRowId = warpM * WMMA_M;
    const int pColId = warpN * WMMA_N;

    if (pRowId >= M || pColId >= N) {
        return;
    }
//    printf("pRowId : %d, pColId : %d\n", pRowId,pColId);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> aFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> bFrag;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
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
        if (tidX == 0) {
            printf(" cur kIter = %d\n", kIter);
            printf(" cur aRowId = %d, aColId = %d, bRowId = %d, bColId = %d\n", aRowId, aColId, bRowId, bColId);
        }
        // Bounds checking
        if (aRowId < M && aColId < K && bRowId < K && bColId < N) {
            const auto aOffsetPtr = matrixA + aRowId * lda + aColId;
            const auto bOffsetPtr = matrixB + bRowId * ldb + bColId;

            nvcuda::wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            nvcuda::wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            nvcuda::wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

//#pragma unroll
//    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
//        const int sIdx = pRowId * ldc + pColId + idx;
//
//        cFrag.x[idx] *= matrixS[sIdx];
//    }
//    if (tidX == 4) {
//        printf("\n aFrag.num_elements : %d\n", aFrag.num_elements);
//        for (int idx = 0; idx < aFrag.num_elements; ++idx) {
//            printf(" %f ", static_cast<float>(aFrag.x[idx]));
//        }
//    }

    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
    nvcuda::wmma::store_matrix_sync(pOffsetPtr, cFrag, ldp, nvcuda::wmma::mem_row_major);
}
