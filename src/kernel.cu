#include <cstdio>

#include <mma.h>

#include "kernel.cuh"
#include "wmmaSetting.hpp"
#include "Matrix.hpp"

template<typename T>
__global__ void printData(size_t n, T *a) {
    for (size_t i = 0; i < n; ++i) {
        printf("%f ", static_cast<float>(a[i]));
    }
}

template __global__ void printData<float>(size_t n, float *a);
template __global__ void printData<half>(size_t n, half *a);

__global__ void convertFp32ToFp16(const size_t n, const float *in, half *out) {
    size_t idx = static_cast<size_t> (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
    }
}

const float SPARSITY_BOUND = 1.0f;

template<typename T>
__device__ float calculateMatrixTileSparsity(const int tileM,
                                             const int tileN,
                                             const size_t ld,
                                             const MatrixStorageOrder storageOrder,
                                             const T *matrixPtr) {
    size_t nnzCount = 0;
#pragma unroll
    for (size_t rowIter = 0; rowIter < tileM; ++rowIter) {
#pragma unroll
        for (size_t colIter = 0; colIter < tileN; ++colIter) {
            if (storageOrder == MatrixStorageOrder::row_major) {
                nnzCount += *(matrixPtr + rowIter * ld + colIter) == 0 ? 0 : 1;
            } else {
                nnzCount += *(matrixPtr + colIter * ld + rowIter) == 0 ? 0 : 1;
            }
        }
    }
    const int numValues = tileM * tileN;
    return static_cast<float>(numValues - nnzCount) / static_cast<float>(numValues);
}

__device__ void matrixTileMultiplicationUseCudaCode(int pRowId, int pColId,
                                                    const size_t M, const size_t N, const size_t K,
                                                    const half *matrixA,
                                                    const half *matrixB,
                                                    const float *matrixS,
                                                    float *matrixP) {

}

__device__ void matrixTileMultiplicationUseTensorCore(int pRowId, int pColId,
                                                      const size_t M, const size_t N, const size_t K,
                                                      const half *matrixA,
                                                      const half *matrixB,
                                                      const float *matrixS,
                                                      float *matrixP) {

    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldp = N;
    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;

//    printf("pRowId : %d, pColId : %d\n", pRowId,pColId);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, nvcuda::wmma::row_major>
        aFrag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, nvcuda::wmma::row_major>
        bFrag;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;
    fill_fragment(cFrag, 0.0f);

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRowId = pRowId;
        const int aColId = kIter;

        const int bRowId = kIter;
        const int bColId = pColId;
//        if (tidX == 0) {
//            printf(" cur kIter = %d\n", kIter);
//            printf(" cur aRowId = %d, aColId = %d, bRowId = %d, bColId = %d\n", aRowId, aColId, bRowId, bColId);
//        }
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

//    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> sFrag;
//    const int lds = N;
//    const auto ptrS = matrixS + pRowId * lds + pColId;
//    nvcuda::wmma::load_matrix_sync(sFrag,ptrS,lds,nvcuda::wmma::mem_row_major);
    if ((blockDim.x * blockIdx.x + threadIdx.x) == 9) {
        printf("\n cFrag.num_elements : %d\n", cFrag.num_elements);
        for (int idx = 0; idx < cFrag.num_elements; ++idx) {
            printf(" %f ", static_cast<float>(cFrag.x[idx]));
        }


//        printf("\n sFrag.num_elements : %d\n", sFrag.num_elements);
//        for (int idx = 0; idx < sFrag.num_elements; ++idx) {
//            printf(" %f ", static_cast<float>(sFrag.x[idx]));
//        }
    }

    nvcuda::wmma::store_matrix_sync(pOffsetPtr, cFrag, ldp, nvcuda::wmma::mem_row_major);
}

__global__ void comp_sddmm_gpu(const size_t M, const size_t N, const size_t K,
                               const half *matrixA, const half *matrixB,
                               const float *matrixS,
                               float *matrixP) {
    const size_t tidX = (blockDim.x * blockIdx.x + threadIdx.x);
    const size_t tidY = (blockDim.y * blockIdx.y + threadIdx.y);

    const size_t warpM = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const size_t warpN = (blockDim.y * blockIdx.y + threadIdx.y);

    // Compute dense matrix multiplication using Tensor core

    const size_t pRowId = warpM * WMMA_M;
    const size_t pColId = warpN * WMMA_N;

    if (pRowId >= M || pColId >= N) {
        return;
    }
    matrixTileMultiplicationUseTensorCore(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    const int ldp = N;
//    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
//    const float sparsity = calculateMatrixTileSparsity(WMMA_M, WMMA_N, ldp, MatrixStorageOrder::row_major, pOffsetPtr);
//    if (sparsity < 0) {
//        matrixTileMultiplicationUseCudaCode(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    } else {
//        matrixTileMultiplicationUseTensorCore(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    }

}
