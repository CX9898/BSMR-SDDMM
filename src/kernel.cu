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

    nvcuda::wmma::store_matrix_sync(pOffsetPtr, cFrag, ldp, nvcuda::wmma::mem_row_major);
}

__device__ void positionCalculator(const size_t tileRow, const size_t tileCol,
                                   const size_t row, const size_t col,
                                   int &laneId, int &idx) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        laneId = -1;
        idx = -1;
        return;
    }
    const int localRow = static_cast<int>(row - tileRow);
    const int localCol = static_cast<int>(col - tileCol);

    const int numberOfIterations = localCol % 8;

    const int startLane = (localRow % 8) * 4;
    laneId = startLane + numberOfIterations / 2;

    const int addNum = numberOfIterations % 2;
    if (localCol < 8) { // idx : 0~3
        if (localRow < 8) { //  idx : 0~1 || 4~5
            idx = addNum;
        } else { // idx : 2~3 || 6~7
            idx = 2 + addNum;
        }
    } else { // idx : 4~7
        if (localRow < 8) { //  idx : 0~1 || 4~5
            idx = 4 + addNum;
        } else { // idx : 2~3 || 6~7
            idx = 6 + addNum;
        }
    }

}

__device__ void matrixTileMultiplicationUseTensorCore2(const size_t pRowId, const size_t pColId,
                                                       const size_t M, const size_t N, const size_t K, const size_t nnz,
                                                       const half *matrixA,
                                                       const half *matrixB,
                                                       const size_t *matrixSRowIndex,
                                                       const size_t *matrixSColIndex,
                                                       const float *matrixS,
                                                       float *matrixP) {
    const size_t tidX = (blockDim.x * blockIdx.x + threadIdx.x);
//    const size_t tidY = (blockDim.y * blockIdx.y + threadIdx.y);
//
//    const size_t warpM = tidX / WARP_SIZE;
//    const size_t warpN = tidY;
//
//    const int landIdM = tidX % WARP_SIZE;
//    const int landIdN = tidY % WARP_SIZE;
//

    // Leading dimensions. Packed with no transpositions.
    const size_t lda = K;
    const size_t ldb = N;

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
    const int laneId = static_cast<int>(tidX % WARP_SIZE); // TODO : 没有考虑到 Y 轴的线程, 不安全

    for (int matrixPIdx = 0; matrixPIdx < nnz; ++matrixPIdx) {
        const size_t curRow = matrixSRowIndex[matrixPIdx];
        const size_t curCol = matrixSColIndex[matrixPIdx];

        int findLaneId, findIdx;
        positionCalculator(pRowId, pColId, curRow, curCol, findLaneId, findIdx);

        if (laneId == findLaneId) {
            matrixP[matrixPIdx] = cFrag.x[findIdx];
//            printf(
//                " pRowId = %d, pColId = %d, curRow = %d, curCol = %d, findLaneId = %d, findIdx = %d, cFrag.x[%d] = %f\n",
//                static_cast<int>(pRowId),
//                static_cast<int>(pColId),
//                static_cast<int>(curRow),
//                static_cast<int>(curCol),
//                findLaneId,
//                findIdx,
//                findIdx,
//                static_cast<float>(cFrag.x[findIdx]));
        }
    }




//    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0) {
////        printf("\n cFrag.num_elements : %d\n", cFrag.num_elements);
//        printf("|T%d|", blockDim.x * blockIdx.x + threadIdx.x);
//        for (int idx = 0; idx < cFrag.num_elements; ++idx) {
////            printf(" %f",
////                   static_cast<float>(cFrag.x[idx]));
//            printf("[%d,%d]|", (laneId & 0b1) + (idx & 0b10), (idx & 0b100) + (laneId & 0b10) + (idx & 0b1));
//
//        }
//        printf("\n");
//
////        printf("\n sFrag.num_elements : %d\n", sFrag.num_elements);
////        for (int idx = 0; idx < sFrag.num_elements; ++idx) {
////            printf(" %f ", static_cast<float>(sFrag.x[idx]));
////        }
//    }

}

__global__ void sddmm_gpu(const size_t M, const size_t N, const size_t K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP) {
    const size_t tidX = (blockDim.x * blockIdx.x + threadIdx.x);
    const size_t tidY = (blockDim.y * blockIdx.y + threadIdx.y);

    const size_t warpM = tidX / WARP_SIZE;
    const size_t warpN = tidY;

//    const int landIdM = tidX % WARP_SIZE;
//    const int landIdN = tidY % WARP_SIZE;

    // Compute dense matrix multiplication using Tensor core

//    const size_t pRowId = warpM * WMMA_M;
//    const size_t pColId = warpN * WMMA_N;
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

__global__ void sddmm_coo_gpu(const size_t M, const size_t N, const size_t K, const size_t nnz,
                              const half *matrixA, const half *matrixB,
                              const size_t *matrixSRowIndex,
                              const size_t *matrixSColIndex,
                              const float *matrixS,
                              float *matrixP) {
    const size_t tidX = (blockDim.x * blockIdx.x + threadIdx.x);
    const size_t tidY = (blockDim.y * blockIdx.y + threadIdx.y);

    const size_t warpM = tidX / WARP_SIZE;
    const size_t warpN = tidY;

//    const int landIdM = tidX % WARP_SIZE;
//    const int landIdN = tidY % WARP_SIZE;

    // Compute dense matrix multiplication using Tensor core

    const size_t pRowId = warpM * WMMA_M;
    const size_t pColId = warpN * WMMA_N;
//    const size_t pRowId = warpN * WMMA_N;
//    const size_t pColId = warpM * WMMA_M;

    if (pRowId >= M || pColId >= N) {
        return;
    }
    matrixTileMultiplicationUseTensorCore2(pRowId,
                                           pColId,
                                           M,
                                           N,
                                           K,
                                           nnz,
                                           matrixA,
                                           matrixB,
                                           matrixSRowIndex,
                                           matrixSColIndex,
                                           matrixS,
                                           matrixP);
//    const int ldp = N;
//    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
//    const float sparsity = calculateMatrixTileSparsity(WMMA_M, WMMA_N, ldp, MatrixStorageOrder::row_major, pOffsetPtr);
//    if (sparsity < 0) {
//        matrixTileMultiplicationUseCudaCode(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    } else {
//        matrixTileMultiplicationUseTensorCore(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    }

}
