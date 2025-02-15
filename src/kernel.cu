#include <cstdio>

#include <mma.h>

#include "kernel.cuh"
#include "TensorCoreConfig.cuh"
#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"

namespace kernel {

using namespace nvcuda;

__global__ void checkFragmentData() {
    const UIN wmmaM = 16;
    const UIN wmmaN = 16;
    const UIN wmmaK = 16;
    const UIN aTileSize = wmmaM * wmmaK;
    const UIN bTileSize = wmmaK * wmmaN;
    __shared__ half aTileSMEM[aTileSize];
    __shared__ half bTileSMEM[bTileSize];

    const UIN warpId = threadIdx.x / WARP_SIZE;
    const UIN laneId = threadIdx.x % WARP_SIZE;

    if (warpId == 0 && laneId == 0) {
        for (int i = 0; i < aTileSize; ++i) {
            aTileSMEM[i] = static_cast<half>(i);

        }
        for (int i = 0; i < bTileSize; ++i) {
            bTileSMEM[i] = static_cast<half>(i);
        }
    }

    if (warpId == 0) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

        fill_fragment(cFrag, 0.0f);

        wmma::load_matrix_sync(aFrag, aTileSMEM, 16);
        wmma::load_matrix_sync(bFrag, bTileSMEM, 16);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        if (laneId == 0) {
            printf("Fragment data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx) {
            if (warpId == 0 && laneId == laneIdx) {
                printf("laneId = %d : ", laneId);
                for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                    printf("%.0f ", cFrag.x[idxOfFragment]);
                }
                printf("\n");
            }
        }
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

// m16n16k16
// blockDim: [64, 1, 1]
__global__ void sddmm_gpu_rebell_m16n16k16_matrixA_row_matrixB_row(const UIN M,
                                                                   const UIN N,
                                                                   const UIN K,
                                                                   const half *matrixA,
                                                                   const half *matrixB,
                                                                   const UIN numNonZeroRow,
                                                                   const UIN *reorderedRows,
                                                                   const UIN *reorderedCols,
                                                                   const UIN *reorderedColOffset,
                                                                   const UIN *blockRowOffsets,
                                                                   const UIN *blockValues,
                                                                   float *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ half aTileSMEM[aTileSMEMSize];
    __shared__ half bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = N;

    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        // Loop over K
        for (int kIter = 0; kIter < K; kIter += WMMA_K) {
            // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
                const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
                const UIN aColId = kIter + laneId % 16;

                aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
            }

            // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const UIN bRowId = kIter + warpId * 8 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<half>(0);
            }
            __syncthreads();

            // Compute the matrix multiplication
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM, WMMA_N);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N, WMMA_N * 2);
                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }

            __syncthreads();
        }

        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
__global__ void sddmm_gpu_rebell_m16n16k16_matrixA_row_matrixB_col(const UIN M,
                                                                   const UIN N,
                                                                   const UIN K,
                                                                   const half *matrixA,
                                                                   const half *matrixB,
                                                                   const UIN numNonZeroRow,
                                                                   const UIN *reorderedRows,
                                                                   const UIN *reorderedCols,
                                                                   const UIN *reorderedColOffset,
                                                                   const UIN *blockRowOffsets,
                                                                   const UIN *blockValues,
                                                                   float *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ half aTileSMEM[aTileSMEMSize];
    __shared__ half bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = K;

    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;

        // Loop over K
        for (int kIter = 0; kIter < K; kIter += WMMA_K) {
            // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 4; ++iter) {
                const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
                const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
                const UIN aColId = kIter + laneId % 16;

                aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
            }

            // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const UIN bRowId = kIter + warpId * 8 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<half>(0);
            }
            __syncthreads();

            // Compute the matrix multiplication
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM, WMMA_K);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N, WMMA_N * 2);
                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }

            __syncthreads();
        }

        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
// 在外部进行K迭代
__global__ void sddmm_gpu_rebell_m16n16k16_out_kIter_matrixA_row_matrixB_row(const UIN M,
                                                                             const UIN N,
                                                                             const UIN K,
                                                                             const UIN kIter,
                                                                             const half *matrixA,
                                                                             const half *matrixB,
                                                                             const UIN numNonZeroRow,
                                                                             const UIN *reorderedRows,
                                                                             const UIN *reorderedCols,
                                                                             const UIN *reorderedColOffset,
                                                                             const UIN *blockRowOffsets,
                                                                             const UIN *blockValues,
                                                                             float *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ half aTileSMEM[aTileSMEMSize];
    __shared__ half bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = N;

    // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
        const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
        const UIN aColId = kIter + laneId % 16;

        aTileSMEM[warpId * 128 + iter * 32 + laneId] =
            (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
    }

    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
        const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const UIN bRowId = kIter + warpId * 8 + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<half>(0);
        }
        __syncthreads();

        // Compute the matrix multiplication
        if (colBlockId < numColBlocksCurrentRowPanel) {
            wmma::load_matrix_sync(aFrag, aTileSMEM, WMMA_N);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N, WMMA_N * 2);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        __syncthreads();


        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] += cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
// 在外部进行K迭代
__global__ void sddmm_gpu_rebell_m16n16k16_out_kIter_matrixA_row_matrixB_col(const UIN M,
                                                                             const UIN N,
                                                                             const UIN K,
                                                                             const UIN kIter,
                                                                             const half *matrixA,
                                                                             const half *matrixB,
                                                                             const UIN numNonZeroRow,
                                                                             const UIN *reorderedRows,
                                                                             const UIN *reorderedCols,
                                                                             const UIN *reorderedColOffset,
                                                                             const UIN *blockRowOffsets,
                                                                             const UIN *blockValues,
                                                                             float *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ half aTileSMEM[aTileSMEMSize];
    __shared__ half bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = K;

    // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
        const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
        const UIN aColId = kIter + laneId % 16;

        aTileSMEM[warpId * 128 + iter * 32 + laneId] =
            (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
    }

    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
        const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const UIN bRowId = kIter + warpId * 8 + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<half>(0);
        }
        __syncthreads();

        // Compute the matrix multiplication
        if (colBlockId < numColBlocksCurrentRowPanel) {
            wmma::load_matrix_sync(aFrag, aTileSMEM, WMMA_N);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N, WMMA_N * 2);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        __syncthreads();


        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] += cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
// 一次加载4*WMMA_K个元素
__global__ void sddmm_gpu_rebell_2_m16n16k16_matrixA_row_matrixB_row(const UIN M,
                                                                     const UIN N,
                                                                     const UIN K,
                                                                     const half *matrixA,
                                                                     const half *matrixB,
                                                                     const UIN numNonZeroRow,
                                                                     const UIN *reorderedRows,
                                                                     const UIN *reorderedCols,
                                                                     const UIN *reorderedColOffset,
                                                                     const UIN *blockRowOffsets,
                                                                     const UIN *blockValues,
                                                                     float *matrixP) {
    __shared__ half aTileSMEM[(16 * 16) * 4];
    __shared__ half bTileSMEM[(16 * 32) * 4];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = N;

    const UIN startIndexOfRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;

        // Loop over K
        for (int kIter = 0; kIter < K; kIter += WMMA_K * 4) {
            // Load matrix A into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 16; ++iter) {
                const UIN reorderedRowIndex = startIndexOfRowsCurrentRowPanel + iter;
                const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
                const UIN aColId = warpId * WARP_SIZE + laneId;

                aTileSMEM[warpId * 32 + iter * 64 + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
            }

            // Load matrix B data into shared memory, each thread loads 32 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 32; ++iter) {
                const UIN bRowId = kIter + warpId * 32 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 1024 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<half>(0);
            }
            __syncthreads();

            // Compute the matrix multiplication
            if (colBlockId < numColBlocksCurrentRowPanel) {
                for(int iter = 0; iter < 4; ++iter){
                    wmma::load_matrix_sync(aFrag, aTileSMEM + iter * 16, WMMA_K * 4);
                    wmma::load_matrix_sync(bFrag, (bTileSMEM + warpId * WMMA_N) + iter * 512, WMMA_N * 2);
                    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                }
            }

            __syncthreads();
        }

        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
// 一次加载4*WMMA_K个元素
__global__ void sddmm_gpu_rebell_2_m16n16k16_matrixA_row_matrixB_col(const UIN M,
                                                                     const UIN N,
                                                                     const UIN K,
                                                                     const half *matrixA,
                                                                     const half *matrixB,
                                                                     const UIN numNonZeroRow,
                                                                     const UIN *reorderedRows,
                                                                     const UIN *reorderedCols,
                                                                     const UIN *reorderedColOffset,
                                                                     const UIN *blockRowOffsets,
                                                                     const UIN *blockValues,
                                                                     float *matrixP) {
    __shared__ half aTileSMEM[(16 * 16) * 4];
    __shared__ half bTileSMEM[(16 * 32) * 4];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = K;

    const UIN startIndexOfRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];
    for (int colBlockIter = 0; colBlockIter < numColBlocksCurrentRowPanel; colBlockIter += 2) {

        // Data needs to be reset to zero before calculating the next column block
        fill_fragment(cFrag, 0.0f);

        const UIN colBlockId = colBlockIter + warpId;
        const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

        const UIN startIndexOfReorderedColsCurrentIter =
            reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
        const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

        const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;

        // Loop over K
        for (int kIter = 0; kIter < K; kIter += WMMA_K * 4) {
            // Load matrix A into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 16; ++iter) {
                const UIN reorderedRowIndex = startIndexOfRowsCurrentRowPanel + iter;
                const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
                const UIN aColId = warpId * WARP_SIZE + laneId;

                aTileSMEM[warpId * 32 + iter * 64 + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<half>(0);
            }

            // Load matrix B data into shared memory, each thread loads 32 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 32; ++iter) {
                const UIN bRowId = kIter + warpId * 32 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 1024 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<half>(0);
            }
            __syncthreads();

            // Compute the matrix multiplication
            if (colBlockId < numColBlocksCurrentRowPanel) {
                for(int iter = 0; iter < 4; ++iter){
                    wmma::load_matrix_sync(aFrag, aTileSMEM + iter * 16, WMMA_K * 4);
                    wmma::load_matrix_sync(bFrag, (bTileSMEM + warpId * WMMA_N) + iter * 512, WMMA_N * 2);
                    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                }
            }

            __syncthreads();
        }

        // Store the result
        if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
            for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                UIN localRow, localCol;
                calculateFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

                const UIN idxOfMatrixP =
                    blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

                // Saved when the value is not 0
                if (idxOfMatrixP != NULL_VALUE) {
                    matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
                }
            }
        }
        __syncthreads();
    }
}


} // namespace kernel

void sddmm_gpu_rebell(const Matrix<float> &matrixA,
                      const Matrix<float> &matrixB,
                      const sparseMatrix::CSR<float> &matrixS,
                      const ReBELL &rebell,
                      sparseMatrix::CSR<float> &matrixP,
                      float &time) {

    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType_dev(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType_dev(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        kernel::convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType_dev.data());
        kernel::convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType_dev.data());
    }

    dev::vector<UIN> reorderedRowIndices_dev(rebell.reorderedRows());
    dev::vector<UIN> reorderedColIndices_dev(rebell.reorderedCols());
    dev::vector<UIN> reorderedColIndicesOffset_dev(rebell.reorderedColOffsets());
    dev::vector<UIN> blockRowOffsets_dev(rebell.blockRowOffsets());
    dev::vector<UIN> blockValues_dev(rebell.blockValues());
    dev::vector<float> matrixP_dev(matrixS.nnz());

    dim3 grid, block;
    block.x = 64;
    grid.x = rebell.numRowPanels();

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    if(matrixA.storageOrder() == MatrixStorageOrder::row_major && matrixB.storageOrder() == MatrixStorageOrder::row_major){
        kernel::sddmm_gpu_rebell_m16n16k16_matrixA_row_matrixB_row<<<grid, block>>>(matrixS.row(), matrixS.col(), matrixA.col(),
        matrixA_values_convertedType_dev.data(),
        matrixB_values_convertedType_dev.data(),
        rebell.reorderedRows().size(),
        reorderedRowIndices_dev.data(),
        reorderedColIndices_dev.data(),
        reorderedColIndicesOffset_dev.data(),
        blockRowOffsets_dev.data(),
        blockValues_dev.data(),
        matrixP_dev.data());
    } else if(matrixA.storageOrder() == MatrixStorageOrder::row_major && matrixB.storageOrder() == MatrixStorageOrder::col_major){
        kernel::sddmm_gpu_rebell_m16n16k16_matrixA_row_matrixB_col<<<grid, block>>>(matrixS.row(), matrixS.col(), matrixA.col(),
        matrixA_values_convertedType_dev.data(),
        matrixB_values_convertedType_dev.data(),
        rebell.reorderedRows().size(),
        reorderedRowIndices_dev.data(),
        reorderedColIndices_dev.data(),
        reorderedColIndicesOffset_dev.data(),
        blockRowOffsets_dev.data(),
        blockValues_dev.data(),
        matrixP_dev.data());
    }

    timeCalculator.endClock();

    time = timeCalculator.getTime();

    matrixP.setValues() = d2h(matrixP_dev);
}

// 在外部进行K迭代
void sddmm_gpu_rebell_out_kIter(const Matrix<float> &matrixA,
                                const Matrix<float> &matrixB,
                                const sparseMatrix::CSR<float> &matrixS,
                                const ReBELL &rebell,
                                sparseMatrix::CSR<float> &matrixP,
                                float &time) {

    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType_dev(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType_dev(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        kernel::convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType_dev.data());
        kernel::convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType_dev.data());
    }

    dev::vector<UIN> reorderedRowIndices_dev(rebell.reorderedRows());
    dev::vector<UIN> reorderedColIndices_dev(rebell.reorderedCols());
    dev::vector<UIN> reorderedColIndicesOffset_dev(rebell.reorderedColOffsets());
    dev::vector<UIN> blockRowOffsets_dev(rebell.blockRowOffsets());
    dev::vector<UIN> blockValues_dev(rebell.blockValues());
    dev::vector<float> matrixP_dev(matrixS.nnz());

    dim3 grid, block;
    block.x = 64;
    grid.x = rebell.numRowPanels();

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    // Loop over K
    for (int kIter = 0; kIter < matrixA.col(); kIter += WMMA_K) {
        kernel::sddmm_gpu_rebell_m16n16k16_out_kIter_matrixA_row_matrixB_row<<<grid, block>>>(matrixS.row(), matrixS.col(), matrixA.col(),kIter,
            matrixA_values_convertedType_dev.data(),
            matrixB_values_convertedType_dev.data(),
            rebell.reorderedRows().size(),
            reorderedRowIndices_dev.data(),
            reorderedColIndices_dev.data(),
            reorderedColIndicesOffset_dev.data(),
            blockRowOffsets_dev.data(),
            blockValues_dev.data(),
            matrixP_dev.data());
    }
    timeCalculator.endClock();

    time = timeCalculator.getTime();

    matrixP.setValues() = d2h(matrixP_dev);
}