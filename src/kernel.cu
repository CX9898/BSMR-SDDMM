#include <cstdio>

#include <mma.h>

#include "kernel.cuh"
#include "TensorCoreConfig.cuh"
#include "reordering.hpp"

namespace kernel {

using namespace nvcuda;

__global__ void convertFp32ToFp16(const UIN n, const float *in, half *out) {
    UIN idx = static_cast<UIN> (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
    }
}

template<typename T>
__global__ void convertDataType(const UIN n, const float *in, T *out) {
    const UIN idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(in[idx]);
//        printf("in[%d] = %f, static_cast<float>out[%d] = %f\n", idx, in[idx], idx, static_cast<float>(out[idx]));
    }
}

template __global__ void convertDataType<int>(const UIN n, const float *in, int *out);
template __global__ void convertDataType<float>(const UIN n, const float *in, float *out);
template __global__ void convertDataType<double>(const UIN n, const float *in, double *out);
template __global__ void convertDataType<half>(const UIN n, const float *in, half *out);

const float SPARSITY_BOUND = 1.0f;

template<typename T>
__device__ float calculateMatrixTileSparsity(const int tileM,
                                             const int tileN,
                                             const UIN ld,
                                             const MatrixStorageOrder storageOrder,
                                             const T *matrixPtr) {
    UIN nnzCount = 0;
#pragma unroll
    for (UIN rowIter = 0; rowIter < tileM; ++rowIter) {
#pragma unroll
        for (UIN colIter = 0; colIter < tileN; ++colIter) {
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
                                                    const UIN M, const UIN N, const UIN K,
                                                    const half *matrixA,
                                                    const half *matrixB,
                                                    const float *matrixS,
                                                    float *matrixP) {

}

__device__ void matrixTileMultiplicationUseTensorCore(int pRowId, int pColId,
                                                      const UIN M, const UIN N, const UIN K,
                                                      const half *matrixA,
                                                      const half *matrixB,
                                                      const float *matrixS,
                                                      float *matrixP) {

    // Leading dimensions. Packed with no transpositions.
    const int lda = K;
    const int ldb = N;
    const int ldp = N;
    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;
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

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

//#pragma unroll
//    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
//        const int sIdx = pRowId * ldc + pColId + idx;
//
//        cFrag.x[idx] *= matrixS[sIdx];
//    }

    wmma::store_matrix_sync(pOffsetPtr, cFrag, ldp, wmma::mem_row_major);
}

__device__ void matrixTileMultiplicationUseTensorCore_coo(TensorCoreConfig tensorCoreConfig,
                                                          const UIN pRowId,
                                                          const UIN pColId,
                                                          const UIN M,
                                                          const UIN N,
                                                          const UIN K,
                                                          const UIN nnz,
                                                          const half *matrixA,
                                                          const half *matrixB,
                                                          const UIN *matrixSRowIndex,
                                                          const UIN *matrixSColIndex,
                                                          const UIN *matrixTileIndex,
                                                          const float *matrixS,
                                                          float *matrixP) {

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

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

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int warpId = tensorCoreConfig.globalWarpId();
    const int laneId = tensorCoreConfig.laneId();

    for (int matrixPIdx = matrixTileIndex[warpId];
         matrixPIdx < matrixTileIndex[warpId + 1]; ++matrixPIdx) {
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

//        if (matrixPIdx == 8410 && warpId == 780 && laneId_ == 0) {
//            printf(" warpId = %d\n"
//                   " pRowId = %d, pColId = %d, curRow = %d, curCol = %d, curValue = %f"
//                   " laneId_ = %d findLaneId = %d, findIdx = %d, cFrag.x[%d] = %f\n",
//                   warpId,
//                   static_cast<int>(pRowId),
//                   static_cast<int>(pColId),
//                   static_cast<int>(curRow),
//                   static_cast<int>(curCol),
//                   static_cast<float>(matrixS[matrixPIdx]),
//                   laneId_,
//                   findLaneId,
//                   findIdx,
//                   findIdx,
//                   static_cast<float>(cFrag.x[findIdx]));
//            printf("frag : ");
//            for (int idx = 0; idx < 8; ++idx) {
//                printf("%f ", static_cast<float>(cFrag.x[idx]));
//            }
//            printf("\n");
//        }
        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
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
}

__global__ void sddmm_gpu(const UIN M, const UIN N, const UIN K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP) {
    const UIN tidX = (blockDim.x * blockIdx.x + threadIdx.x);
    const UIN tidY = (blockDim.y * blockIdx.y + threadIdx.y);

    const UIN warpM = tidX / WARP_SIZE;
    const UIN warpN = tidY;

//    const int landIdM = tidX % WARP_SIZE;
//    const int landIdN = tidY % WARP_SIZE;

    // Compute dense matrix multiplication using Tensor core

    const UIN pRowId = warpM * WMMA_M;
    const UIN pColId = warpN * WMMA_N;
//    const UIN pRowId = warpN * WMMA_N;
//    const UIN pColId = warpM * WMMA_M;

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

__global__ void sddmm_gpu_coo_1(TensorCoreConfig tensorCoreConfig,
                                const UIN M, const UIN N, const UIN K, const UIN nnz,
                                const half *matrixA, const half *matrixB,
                                const UIN *matrixSRowIndex,
                                const UIN *matrixSColIndex,
                                const UIN *matrixTileIndex,
                                const float *matrixS,
                                float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    if (pRowId >= M || pColId >= N) {
        return;
    }

    const int warpId = tensorCoreConfig.globalWarpId();
    const int numData = matrixTileIndex[warpId + 1] - matrixTileIndex[warpId];
    if (numData <= 0) {
        return;
    }

    // Compute dense matrix multiplication using Tensor core
    matrixTileMultiplicationUseTensorCore_coo(tensorCoreConfig,
                                              pRowId,
                                              pColId,
                                              M,
                                              N,
                                              K,
                                              nnz,
                                              matrixA,
                                              matrixB,
                                              matrixSRowIndex,
                                              matrixSColIndex,
                                              matrixTileIndex,
                                              matrixS,
                                              matrixP);
//    const int ldp = N;
//    const auto pOffsetPtr = matrixP + pRowId * ldp + pColId;
//    const float sparsity = (WMMA_M * WMMA_N - numData) / WMMA_M * WMMA_N;
//    if (sparsity < 0) {
//        matrixTileMultiplicationUseCudaCode(pRowId, pColId, M, N, K, matrixA, matrixB, matrixS, matrixP);
//    } else {
//        matrixTileMultiplicationUseTensorCore_coo(tensorCoreConfig,
//                                                  pRowId,
//                                                  pColId,
//                                                  M,
//                                                  N,
//                                                  K,
//                                                  nnz,
//                                                  matrixA,
//                                                  matrixB,
//                                                  matrixSRowIndex,
//                                                  matrixSColIndex,
//                                                  matrixTileMappedToWarpIndex,
//                                                  matrixS,
//                                                  matrixP);
//    }

}

__global__ void sddmm_gpu_coo_2(TensorCoreConfig tensorCoreConfig,
                                const UIN M, const UIN N, const UIN K, const UIN nnz,
                                const half *matrixA, const half *matrixB,
                                const UIN *matrixSRowIndex,
                                const UIN *matrixSColIndex,
                                const float *matrixS,
                                const UIN *matrixSTileMappedToWarpIndex,
                                const UIN *matrixSTileMappedToWarpIndexData,
                                float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    if (pRowId >= M || pColId >= N) {
        return;
    }

    const int globalWarpId = tensorCoreConfig.globalWarpId();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];
    const int numData = tileIndexEnd - tileIndexBegin;
    if (numData <= 0) {
        return;
    }

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

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

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int laneId = tensorCoreConfig.laneId();

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = matrixSTileMappedToWarpIndexData[tileIndexDataIdx];
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

//        if (matrixPIdx == 8820 && globalWarpId == 661 && laneId == 0) {
//            printf("matrixPIdx == %d "
//                   " globalWarpId = %d "
//                   " pRowId = %d, pColId = %d, curRow = %d, curCol = %d, curValue = %f"
//                   " laneId = %d findLaneId = %d, findIdx = %d, cFrag.x[%d] = %f\n",
//                   matrixPIdx,
//                   globalWarpId,
//                   static_cast<int>(pRowId),
//                   static_cast<int>(pColId),
//                   static_cast<int>(curRow),
//                   static_cast<int>(curCol),
//                   static_cast<float>(matrixS[matrixPIdx]),
//                   laneId,
//                   fragmentInformation.laneId_,
//                   fragmentInformation.index_,
//                   fragmentInformation.index_,
//                   static_cast<float>(cFrag.x[fragmentInformation.laneId_]));
//        }
        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
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

}

__global__ void bank_conflicts_test(UIN N, UIN K, const int *matrixB, const int *matrixA) {

    int pRowId = 0;
    int pColId = 0;

    int localWarpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ int aTile[MATRIX_TILE_A_SIZE_PER_BLOCK];

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K * NUM_OF_Y_PER_BLOCK) {
        const int aRowId = pRowId;
        const int aColId = kIter;

        const int bRowId = kIter;
        const int bColId = pColId;

        const auto aOffsetPtr = matrixA + aRowId * lda + aColId;
        const auto bOffsetPtr = matrixB + bRowId * ldb + bColId;

        for (int iter = 0; iter < 8; ++iter) {
            int beginIdxOfSharedMemory = localWarpId * NUMBER_OF_MEMORY_ACCESSES_MATRIX_TILE_A_PER_WARP;
            aTile[beginIdxOfSharedMemory + laneId + iter * WARP_SIZE] =
                matrixA[beginIdxOfSharedMemory + laneId + iter * WARP_SIZE];
        }
    }

}

//__device__ void loadMatrixAToSharedMemorySync(UIN laconst half *matrixA, half *sharedMemory) {
//
//#pragma unroll
//    for (int iter = 0; iter < 8; ++iter) {
//        const UIN startIdxOfSharedMemoryOfMtxA = localWarpId * NUMBER_OF_MATRIX_A_TILE_MEMORY_ACCESSES_PER_WARP;
//        const UIN startIdxOfGlobalMemoryOfMtxA = pRowIdForBlock * lda + K;
//        const UIN iterationSpan = laneId + iter * WARP_SIZE;
//        sharedMemory[startIdxOfSharedMemoryOfMtxA + iterationSpan] =
//            matrixA[startIdxOfGlobalMemoryOfMtxA + iterationSpan];
//    }
//    __syncthreads();
//}

// 在核函数中加入共享内存: 整块64×64的矩阵块A和块B按连续的顺序载入共享内存.
// 未完全实现.
// 问题: 共享内存中一次储存64×64个矩阵数据, 但是超过这个大小的矩阵无法载入, 会出现错误
// 放弃原因: 在 `wmma::load_matrix_sync` 中, 出现了bank conflict, 无法解决
__global__ void sddmm_gpu_coo_4_matrixA_row_matrixB_row(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN globalWarpId = tensorCoreConfig.globalWarpId();

    __shared__ half aTileSharedMem[MATRIX_TILE_A_SIZE_PER_BLOCK];
    __shared__ half bTileSharedMem[MATRIX_TILE_B_SIZE_PER_BLOCK];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    const UIN pRowIdForBlock = tensorCoreConfig.blockStarRow();
    const UIN pColIdForBlock = tensorCoreConfig.blockStarCol();

    const UIN localWarpId = tensorCoreConfig.localWarpId();
    const UIN laneId = tensorCoreConfig.laneId();

    const UIN localWarpX = tensorCoreConfig.localWarpX();
    const UIN localWarpY = tensorCoreConfig.localWarpY();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];

    const int numDataInThisWarp = tileIndexEnd - tileIndexBegin;

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K * NUM_OF_WARP_X_PER_BLOCK) {

        // load matrix tile A to shared memory
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {

//            const UIN localARowIdForThisIteration = ;
//            const UIN localAColIdForThisIteration = ;
//            const UIN localBRowIdForThisIteration = ;
//            const UIN localBColIdForThisIteration = ;
//
//            const UIN indexForThisIterationA = laneId + iter * WARP_SIZE;
//            aTileSharedMem[startIdxOfSharedMemoryOfMtxA + indexForThisIterationA] =
//                matrixA[startIdxOfGlobalMemoryOfMtxA + indexForThisIterationA];
//
//            const UIN indexForThisIterationB = laneId + iter * WARP_SIZE;
//            bTileSharedMem[startIdxOfSharedMemoryOfMtxB + indexForThisIterationB] =
//                matrixB[startIdxOfGlobalMemoryOfMtxB + indexForThisIterationB];

//            if (globalWarpId == 0 && localWarpId == 5 && laneId == 0 && kIter == 0) {
//                printf("startIdxOfSharedMemoryOfMtxA + indexForThisIterationA = %d",
//                       startIdxOfSharedMemoryOfMtxA + indexForThisIterationA);
//            }

//            if (static_cast<int>( matrixA[startIdxOfGlobalMemoryOfMtxA + indexForThisIterationA]) == 1024
//                && blockIdx.y == 0 && globalWarpId == 29) {
//                printf(
//                    "globalWarpId = %d, localWarpId = %d, laneId = %d, kIter = %d, startIdxOfSharedMemoryOfMtxA = %d, startIdxOfGlobalMemoryOfMtxA = %d, indexForThisIterationA = %d, matrixA = %d\n",
//                    globalWarpId,
//                    localWarpId,
//                    laneId,
//                    kIter,
//                    startIdxOfSharedMemoryOfMtxA,
//                    startIdxOfGlobalMemoryOfMtxA,
//                    indexForThisIterationA,
//                    static_cast<int>(matrixA[startIdxOfGlobalMemoryOfMtxA + indexForThisIterationA]));
//            }
        }
        __syncthreads();

//        if (blockIdx.x == 0 && blockIdx.y == 0 && localWarpId == 0 && laneId == 0 && kIter == 0) {
//            for (int i = 0; i < MATRIX_TILE_A_SIZE_PER_BLOCK; ++i) {
//                printf("[%d] = %d\n", i, static_cast<int>(aTileSharedMem[i]));
//            }
//        }

        if (numDataInThisWarp > 0) {
            for (int sharedMemIter = 0; sharedMemIter < NUMBER_OF_MATRIX_TILE_K_IN_SHARED_MEMORY; ++sharedMemIter) {
                const UIN localKIterInSharedMem = sharedMemIter * WMMA_K;
                const auto aOffsetPtr = aTileSharedMem
                    + (localWarpY * WMMA_M * MATRIX_TILE_A_LEADING_DIMENSION)
                    + localKIterInSharedMem;
                const auto bOffsetPtr = bTileSharedMem
                    + (localKIterInSharedMem * MATRIX_TILE_B_LEADING_DIMENSION)
                    + localWarpX * WMMA_N;

                wmma::load_matrix_sync(aFrag, aOffsetPtr, MATRIX_TILE_A_LEADING_DIMENSION);
                wmma::load_matrix_sync(bFrag, bOffsetPtr, MATRIX_TILE_B_LEADING_DIMENSION);

                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }
        }
        __syncthreads();
    }

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = tileIndexDataIdx;
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}

// 在核函数中加入共享内存: 整块64×64的矩阵块A和块B按照16×16的块的顺序载入共享内存
__global__ void sddmm_gpu_coo_5_matrixA_row_matrixB_row(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN globalWarpId = tensorCoreConfig.globalWarpId();

    __shared__ half aTileSMEM[MATRIX_TILE_A_SIZE_PER_BLOCK];
    __shared__ half bTileSMEM[MATRIX_TILE_B_SIZE_PER_BLOCK];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    const UIN localWarpId = tensorCoreConfig.localWarpId();
    const UIN laneId = tensorCoreConfig.laneId();

    const UIN localWarpX = tensorCoreConfig.localWarpX();
    const UIN localWarpY = tensorCoreConfig.localWarpY();

    const UIN startIndexOfMatrixS = matrixSTileMappedToWarpIndex[globalWarpId];
    const UIN endIndexOfMatrixS = matrixSTileMappedToWarpIndex[globalWarpId + 1];

    const UIN numDataInThisWarp = endIndexOfMatrixS - startIndexOfMatrixS;

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    const UIN startIndexOfSharedMemoryOfMatrixA = localWarpId * NUMBER_OF_MEMORY_ACCESSES_MATRIX_TILE_A_PER_WARP;
    const UIN startIndexOfSharedMemoryOfMatrixB = localWarpId * NUMBER_OF_MEMORY_ACCESSES_MATRIX_TILE_B_PER_WARP;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += ITERATION_STEP_OF_K) {
        // Load matrix tile A to shared memory
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {

            const UIN localRowIdInThisIteration = 2 * iter + laneId / WMMA_K;
            const UIN localColIdInThisIteration = laneId % WMMA_K;

            const UIN aRowId = pRowId + localRowIdInThisIteration;
            const UIN aColId = kIter + localWarpX * WMMA_K + localColIdInThisIteration;

            const UIN bRowId = kIter + localWarpY * WMMA_K + localRowIdInThisIteration;
            const UIN bColId = pColId + localColIdInThisIteration;

            const UIN indexOfSharedMemoryInThisIteration = iter * WARP_SIZE + laneId;

            aTileSMEM[startIndexOfSharedMemoryOfMatrixA + indexOfSharedMemoryInThisIteration] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);

            bTileSMEM[startIndexOfSharedMemoryOfMatrixB + indexOfSharedMemoryInThisIteration] =
                (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<half>(0);
        }
        __syncthreads();

        // Only warps where data exists need to be computed
        if (numDataInThisWarp > 0) {
            for (int sharedMemIter = 0; sharedMemIter < NUMBER_OF_MATRIX_TILE_K_IN_SHARED_MEMORY; ++sharedMemIter) {
                const auto aOffsetPtr = aTileSMEM
                    + (localWarpY * NUM_OF_WARP_X_PER_BLOCK + sharedMemIter) * MATRIX_TILE_A_SIZE;
                const auto bOffsetPtr = bTileSMEM
                    + (sharedMemIter * NUM_OF_WARP_X_PER_BLOCK + localWarpX) * MATRIX_TILE_B_SIZE;

                wmma::load_matrix_sync(aFrag, aOffsetPtr, WMMA_K);
                wmma::load_matrix_sync(bFrag, bOffsetPtr, WMMA_M);

                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }
        }
        __syncthreads();
    }

    for (UIN matrixPIdx = startIndexOfMatrixS; matrixPIdx < endIndexOfMatrixS; ++matrixPIdx) {
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}

__global__ void sddmm_gpu_coo_3_matrixA_row_matrixB_row(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN globalWarpId = tensorCoreConfig.globalWarpId();

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];

    __shared__ half aTile[WMMA_M * NUM_OF_Y_PER_BLOCK * WMMA_K * NUM_OF_WARP_X_PER_BLOCK];
    __shared__ half bTile[WMMA_K * NUM_OF_Y_PER_BLOCK * WMMA_N * NUM_OF_WARP_X_PER_BLOCK];

    const int numData = tileIndexEnd - tileIndexBegin;
    if (numData <= 0) {
        return;
    }

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

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

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int laneId = tensorCoreConfig.laneId();

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = tileIndexDataIdx;
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}

__global__ void sddmm_gpu_coo_3_matrixA_row_matrixB_col(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    if (pRowId >= M || pColId >= N) {
        return;
    }

    const int globalWarpId = tensorCoreConfig.globalWarpId();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];
    const int numData = tileIndexEnd - tileIndexBegin;
    if (numData <= 0) {
        return;
    }

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = K;
    const UIN ldb = K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

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
            const auto bOffsetPtr = matrixB + bRowId + bColId * ldb;

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int laneId = tensorCoreConfig.laneId();

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = tileIndexDataIdx;
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}


// TODO：Finish the following kernels. Error occurs when compiling the code.
__global__ void sddmm_gpu_coo_3_matrixA_col_matrixB_row(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    if (pRowId >= M || pColId >= N) {
        return;
    }

    const int globalWarpId = tensorCoreConfig.globalWarpId();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];
    const int numData = tileIndexEnd - tileIndexBegin;
    if (numData <= 0) {
        return;
    }

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = M;
    const UIN ldb = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::col_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRowId = pRowId;
        const int aColId = kIter;

        const int bRowId = kIter;
        const int bColId = pColId;

        // Bounds checking
        if (aRowId < M && aColId < K && bRowId < K && bColId < N) {
            const auto aOffsetPtr = matrixA + aRowId + aColId * lda;
            const auto bOffsetPtr = matrixB + bRowId * ldb + bColId;

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int laneId = tensorCoreConfig.laneId();

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = tileIndexDataIdx;
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}

// TODO: Finish the following kernels. Error occurs when compiling the code.
__global__ void sddmm_gpu_coo_3_matrixA_col_matrixB_col(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP) {
    tensorCoreConfig.initByKernel(blockIdx, blockDim, threadIdx);

    const UIN pRowId = tensorCoreConfig.warpStarRow();
    const UIN pColId = tensorCoreConfig.warpStarCol();

    if (pRowId >= M || pColId >= N) {
        return;
    }

    const int globalWarpId = tensorCoreConfig.globalWarpId();

    const int tileIndexBegin = matrixSTileMappedToWarpIndex[globalWarpId];
    const int tileIndexEnd = matrixSTileMappedToWarpIndex[globalWarpId + 1];
    const int numData = tileIndexEnd - tileIndexBegin;
    if (numData <= 0) {
        return;
    }

    // Leading dimensions. Packed with no transpositions.
    const UIN lda = M;
    const UIN ldb = K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::col_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRowId = pRowId;
        const int aColId = kIter;

        const int bRowId = kIter;
        const int bColId = pColId;

        // Bounds checking
        if (aRowId < M && aColId < K && bRowId < K && bColId < N) {
            const auto aOffsetPtr = matrixA + aRowId + aColId * lda;
            const auto bOffsetPtr = matrixB + bRowId + bColId * ldb;

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    const int laneId = tensorCoreConfig.laneId();

    for (int tileIndexDataIdx = tileIndexBegin; tileIndexDataIdx < tileIndexEnd; ++tileIndexDataIdx) {
        const UIN matrixPIdx = tileIndexDataIdx;
        const UIN curRow = matrixSRowIndex[matrixPIdx];
        const UIN curCol = matrixSColIndex[matrixPIdx];

        FragmentInformation fragmentInformation;
        tensorCoreConfig.positionCalculator(pRowId, pColId, curRow, curCol, fragmentInformation);

        if (laneId == fragmentInformation.laneId_) {
            matrixP[matrixPIdx] = cFrag.x[fragmentInformation.index_];
        }
    }
}

// blockDim: [64, 1, 1]
__global__ void sddmm_gpu_csr_matrix_row_matrix_row(const UIN M,
                                                    const UIN N,
                                                    const UIN K,
                                                    const half *matrixA,
                                                    const half *matrixB,
                                                    const UIN numNonZeroRow,
                                                    const UIN *reorderedMatrixRowIndices,
                                                    const UIN *reorderedMatrixColIndicesOffset,
                                                    const UIN *reorderedMatrixColIndicesInEachRowPanel,
                                                    const UIN *reorderedMatrixPanelOffsets,
                                                    float *matrixP) {
    __shared__ half aTileSMEM[256];
    __shared__ half bTileSMEM[256];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x % WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = N;

    const UIN numPanels = reorderedMatrixPanelOffsets[rowPanelId + 1] - reorderedMatrixPanelOffsets[rowPanelId];
    for (int colTileIdx = 0; colTileIdx < numPanels; colTileIdx += 2) {
        const UIN startIndexOfColTile =
            reorderedMatrixColIndicesOffset[rowPanelId] + col_tile_size * colTileIdx;
        const UIN endIndexOfColTile = reorderedMatrixColIndicesOffset[rowPanelId + 1];

        // Loop over K
        for (int kIter = 0; kIter < K; kIter += WMMA_K) {
            // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                const UIN idxOfReorderedMatrixRowIndices =
                    (rowPanelId * row_panel_size) + (warpId * 8) + (laneId / 16) + (iter * 2);
                const UIN aRowId = idxOfReorderedMatrixRowIndices < numNonZeroRow ?
                    reorderedMatrixRowIndices[idxOfReorderedMatrixRowIndices] : M;
                const UIN aColId = kIter + laneId;

                aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
            }

            // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const UIN bRowId = warpId * 8 + iter;
                const UIN idxOfReorderedMatrixColIndicesInEachRowPanel = startIndexOfColTile + laneId;
                const UIN bColId = idxOfReorderedMatrixColIndicesInEachRowPanel < endIndexOfColTile ?
                    reorderedMatrixColIndicesInEachRowPanel[idxOfReorderedMatrixColIndicesInEachRowPanel] : N;

                bTileSMEM[warpId * 256 + laneId * 32] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<half>(0);
            }

            __syncthreads();

            // Compute the matrix multiplication
            {
                wmma::load_matrix_sync(aFrag, aTileSMEM, WMMA_N);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N, WMMA_N * 2);

                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }

            __syncthreads();
        }

        // Store the result

    }
}

} // namespace kernel

void calculateKernelSettings(const UIN size, UIN &numBlocks, UIN &numThreads) {
    const UIN maxThreadsPerBlock = 1024;
    numThreads = size < maxThreadsPerBlock ? size : maxThreadsPerBlock;
    numBlocks = (size + numThreads - 1) / numThreads;
}

void sddmm_gpu_coo_3(TensorCoreConfig tensorCoreConfig,
                     const UIN M, const UIN N, const UIN K,
                     const half *matrixA, const MatrixStorageOrder matrixAStorageOrder,
                     const half *matrixB, const MatrixStorageOrder matrixBStorageOrder,
                     const UIN *matrixSRowIndex,
                     const UIN *matrixSColIndex,
                     const float *matrixS,
                     const UIN *matrixSTileMappedToWarpIndex,
                     float *matrixP) {
    if (matrixAStorageOrder == MatrixStorageOrder::row_major && matrixBStorageOrder == MatrixStorageOrder::row_major) {
        kernel::sddmm_gpu_coo_5_matrixA_row_matrixB_row<<<tensorCoreConfig.gridDim(), tensorCoreConfig.blockDim()>>>(tensorCoreConfig,
            M,
            N,
            K,
            matrixA,
            matrixB,
            matrixSRowIndex,
            matrixSColIndex,
            matrixS,
            matrixSTileMappedToWarpIndex,
            matrixP);
    } else if (matrixAStorageOrder == MatrixStorageOrder::row_major
        && matrixBStorageOrder == MatrixStorageOrder::col_major) {
        kernel::sddmm_gpu_coo_3_matrixA_row_matrixB_col<<<tensorCoreConfig.gridDim(), tensorCoreConfig.blockDim()>>>(tensorCoreConfig,
            M,
            N,
            K,
            matrixA,
            matrixB,
            matrixSRowIndex,
            matrixSColIndex,
            matrixS,
            matrixSTileMappedToWarpIndex,
            matrixP);
    } else if (matrixAStorageOrder == MatrixStorageOrder::col_major
        && matrixBStorageOrder == MatrixStorageOrder::row_major) {
        kernel::sddmm_gpu_coo_3_matrixA_col_matrixB_row<<<tensorCoreConfig.gridDim(), tensorCoreConfig.blockDim()>>>(tensorCoreConfig,
            M,
            N,
            K,
            matrixB,
            matrixA,
            matrixSColIndex,
            matrixSRowIndex,
            matrixS,
            matrixSTileMappedToWarpIndex,
            matrixP);
    } else {
        kernel::sddmm_gpu_coo_3_matrixA_col_matrixB_col<<<tensorCoreConfig.gridDim(), tensorCoreConfig.blockDim()>>>(tensorCoreConfig,
            M,
            N,
            K,
            matrixB,
            matrixA,
            matrixSColIndex,
            matrixSRowIndex,
            matrixS,
            matrixSTileMappedToWarpIndex,
            matrixP);
    }

}