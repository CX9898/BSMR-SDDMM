#include <cstdio>

#include <mma.h>

#include "cudaUtil.cuh"
#include "sddmmKernel.cuh"
#include "TensorCoreConfig.cuh"
#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"

namespace kernel {

using namespace nvcuda;

__global__ void checkFragmentData() {
    constexpr UIN wmmaM = 16;
    constexpr UIN wmmaN = 16;
    constexpr UIN wmmaK = 8;
    using matrixAType = float;
    using matrixBType = float;
    using matrixATypeFragment = wmma::precision::tf32;
    using matrixBTypeFragment = wmma::precision::tf32;

    constexpr UIN aTileSize = wmmaM * wmmaK;
    constexpr UIN bTileSize = wmmaK * wmmaN;

    constexpr UIN bRow = wmmaN;
    constexpr UIN bCol = wmmaK;

    constexpr UIN ldATile = wmmaK;
    constexpr UIN ldBTile = wmmaK;

    __shared__ matrixAType aTileSMEM[aTileSize];
    __shared__ matrixBType bTileSMEM[bTileSize];

    const UIN warpId = threadIdx.x / WARP_SIZE;
    const UIN laneId = threadIdx.x % WARP_SIZE;

    if (warpId == 0 && laneId == 0) {
        for (int i = 0; i < aTileSize; ++i) {
            aTileSMEM[i] = static_cast<matrixAType>(i);

        }

//        int row = 0;
//        int col = 0;
//        for (int i = 0; i < bTileSize; ++i) {
//            row %= wmmaK;
//            bTileSMEM[i] = static_cast<matrixBType>(row * wmmaK + col);
//            ++row;
//            if (i % ldBTile == 0) {
//                ++col;
//            }
//        }
        if (bRow == wmmaK) {
            for (int i = 0; i < bTileSize; ++i) {
                bTileSMEM[i] = static_cast<matrixBType>(i);
            }
        } else {
            for (int row = 0; row < wmmaK; ++row) {
                for (int col = 0; col < wmmaN; ++col) {
                    bTileSMEM[row + col * ldBTile] = static_cast<matrixBType>(row * wmmaN + col);
                }
            }
        }
    }

    if (warpId == 0 && laneId == 0) {
        printf("\nmatrix A data : \n");
        printf("| |");
        for (int col = 0; col < wmmaK; ++col) {
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < wmmaK + 1; ++i) {
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < wmmaM; ++row) {
            printf("|%d|", row);
            for (int col = 0; col < wmmaK; ++col) {
                printf("%.0f|", static_cast<float>(aTileSMEM[row * wmmaK + col]));
            }
            printf("\n");
        }

        printf("\nmatrix B data : ");
        if (ldBTile == wmmaN) { printf("(rwo major)\n"); } else { printf("(column major)\n"); }
        printf("| |");
        for (int col = 0; col < bCol; ++col) {
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < bCol + 1; ++i) {
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < bRow; ++row) {
            printf("|%d|", row);
            for (int col = 0; col < bCol; ++col) {
                printf("%.0f|", static_cast<float>(bTileSMEM[row * ldBTile + col]));
            }
            printf("\n");
        }
        printf("\n");

        printf("\nmatrix C data : \n");
        printf("| |");
        for (int col = 0; col < wmmaN; ++col) {
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < wmmaN + 1; ++i) {
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < wmmaM; ++row) {
            printf("|%d|", row);
            for (int col = 0; col < wmmaN; ++col) {
                float c = 0.0f;
                for (int k = 0; k < wmmaK; ++k) {
                    const float a = aTileSMEM[row * ldATile + k];
                    const float b = bTileSMEM[k + col * ldBTile];
                    c += a * b;
                }
                printf("%.0f|", static_cast<float>(c));
            }
            printf("\n");
        }
        printf("\n");
    }

    if (warpId == 0) {
        wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, matrixATypeFragment, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, matrixBTypeFragment, wmma::col_major> bFrag;

        wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> cFrag;

        fill_fragment(cFrag, 0.0f);

        wmma::load_matrix_sync(aFrag, aTileSMEM, ldATile);
        wmma::load_matrix_sync(bFrag, bTileSMEM, ldBTile);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        if (laneId == 0) {
            printf("\nFragment A tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx) {
            if (warpId == 0 && laneId == laneIdx) {
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < aFrag.num_elements; ++idxOfFragment) {
                    printf("%.0f|", static_cast<float>(aFrag.x[idxOfFragment]));
                }
                printf("\n");
            }
        }

        if (laneId == 0) {
            printf("\nFragment B tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx) {
            if (warpId == 0 && laneId == laneIdx) {
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < bFrag.num_elements; ++idxOfFragment) {
                    printf("%.0f|", static_cast<float>(bFrag.x[idxOfFragment]));
                }
                printf("\n");
            }
        }

        if (laneId == 0) {
            printf("\nFragment C tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx) {
            if (warpId == 0 && laneId == laneIdx) {
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {
                    printf("%.0f|", static_cast<float>(cFrag.x[idxOfFragment]));
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
__global__ void sddmm_gpu_dense_block_m16n16k16_block64_rowPanel_matrixA_rowMaj_matrixB_rowMaj(const UIN M,
                                                                                               const UIN N,
                                                                                               const UIN K,
                                                                                               const MATRIX_A_TYPE *matrixA,
                                                                                               const MATRIX_B_TYPE *matrixB,
                                                                                               const UIN numNonZeroRow,
                                                                                               const UIN *reorderedRows,
                                                                                               const UIN *reorderedCols,
                                                                                               const UIN *reorderedColOffset,
                                                                                               const UIN *blockRowOffsets,
                                                                                               const UIN *blockValues,
                                                                                               MATRIX_C_TYPE *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::row_major> bFrag;

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
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
            }

            // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const UIN bRowId = kIter + warpId * 8 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<MATRIX_B_TYPE>(0);
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
                calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

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
// 一个thread block负责一个row panel
__global__ void sddmm_gpu_dense_block_m16n16k16_block64_rowPanel_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                               const UIN N,
                                                                                               const UIN K,
                                                                                               const MATRIX_A_TYPE *matrixA,
                                                                                               const MATRIX_B_TYPE *matrixB,
                                                                                               const UIN numNonZeroRow,
                                                                                               const UIN *reorderedRows,
                                                                                               const UIN *reorderedCols,
                                                                                               const UIN *reorderedColOffset,
                                                                                               const UIN *blockRowOffsets,
                                                                                               const UIN *blockValues,
                                                                                               MATRIX_C_TYPE *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::row_major> bFrag;

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
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
            }

            // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
#pragma unroll
            for (int iter = 0; iter < 8; ++iter) {
                const UIN bRowId = kIter + warpId * 8 + iter;
                const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                    reorderedCols[reorderedColIndex] : N;

                bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0);
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
                calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

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
// 一个thread block负责一个row panel中的2个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block64_matrixA_rowMaj_matrixB_rowMaj(const UIN M,
                                                                                      const UIN N,
                                                                                      const UIN K,
                                                                                      const MATRIX_A_TYPE *matrixA,
                                                                                      const MATRIX_B_TYPE *matrixB,
                                                                                      const UIN numNonZeroRow,
                                                                                      const UIN *reorderedRows,
                                                                                      const UIN *reorderedCols,
                                                                                      const UIN *reorderedColOffset,
                                                                                      const UIN *blockRowOffsets,
                                                                                      const UIN *blockValues,
                                                                                      MATRIX_C_TYPE *matrixP) {
    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentIter = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;

    const UIN lda = K;
    const UIN ldb = N;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId % 16;

            aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
        }

        // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const UIN bRowId = kIter + warpId * 8 + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<MATRIX_B_TYPE>(0);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k16
// blockDim: [64, 1, 1]
// 一个thread block负责一个row panel中的2个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block64_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                      const UIN N,
                                                                                      const UIN K,
                                                                                      const MATRIX_A_TYPE *matrixA,
                                                                                      const MATRIX_B_TYPE *matrixB,
                                                                                      const UIN numNonZeroRow,
                                                                                      const UIN *reorderedRows,
                                                                                      const UIN *reorderedCols,
                                                                                      const UIN *reorderedColOffset,
                                                                                      const UIN *blockRowOffsets,
                                                                                      const UIN *blockValues,
                                                                                      MATRIX_C_TYPE *matrixP) {
    constexpr int eachThreadBlockCountsTheNumberOfColBlocks = 2;

    constexpr int aTileSMEMSize = WMMA_M * WMMA_N;
    constexpr int bTileSMEMSize = WMMA_K * WMMA_N * eachThreadBlockCountsTheNumberOfColBlocks;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x % WARP_SIZE;
    const UIN warpId = threadIdx.x / WARP_SIZE;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * eachThreadBlockCountsTheNumberOfColBlocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentIter = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN reorderedColIndex = startIndexOfReorderedColsCurrentIter + laneId;

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 8) + (laneId / 16) + (iter * 2);
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId % 16;

            aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
        }

        // Load matrix B data into shared memory, each thread loads 8 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const UIN bRowId = kIter + warpId * 8 + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 256 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k16
// blockDim: [128, 1, 1]
// 一个thread block负责一个row panel中的4个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block128_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                       const UIN N,
                                                                                       const UIN K,
                                                                                       const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                       const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                       const UIN numNonZeroRow,
                                                                                       const UIN *__restrict__ reorderedRows,
                                                                                       const UIN *__restrict__ reorderedCols,
                                                                                       const UIN *__restrict__ reorderedColOffset,
                                                                                       const UIN *__restrict__ blockRowOffsets,
                                                                                       const UIN *__restrict__ blockValues,
                                                                                       MATRIX_C_TYPE *matrixP) {
    constexpr int eachThreadBlockCountsTheNumberOfColBlocks = 4;

    constexpr int aTileSMEMSize = (WMMA_M * WMMA_N) * 2;
    constexpr int bTileSMEMSize = (WMMA_K * WMMA_N * eachThreadBlockCountsTheNumberOfColBlocks) * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * eachThreadBlockCountsTheNumberOfColBlocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K * 2) {
        // Load matrix A into shared memory, each thread loads 4 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 4) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[warpId * 128 + iter * 32 + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
        }

        // Load matrix B data into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const UIN bRowId = kIter + laneId;
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 512 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0);
        }

        __syncthreads();

        // Compute the matrix multiplication
        for (int iter = 0; iter < 2; ++iter) {
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter * WMMA_K, WMMA_K * 2);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * 512 + iter * WMMA_K, WMMA_K * 2);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k16
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block256_matrixA_rowMaj_matrixB_rowMaj(const UIN M,
                                                                                       const UIN N,
                                                                                       const UIN K,
                                                                                       const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                       const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                       const UIN numNonZeroRow,
                                                                                       const UIN *__restrict__ reorderedRows,
                                                                                       const UIN *__restrict__ reorderedCols,
                                                                                       const UIN *__restrict__ reorderedColOffset,
                                                                                       const UIN *__restrict__ blockRowOffsets,
                                                                                       const UIN *__restrict__ blockValues,
                                                                                       MATRIX_C_TYPE *matrixP) {
    constexpr int number_of_tiles_loaded_in_one_cycle = 32 / WMMA_K;

    constexpr int aTileSMEMSize = (WMMA_M * WMMA_K) * number_of_tiles_loaded_in_one_cycle;
    constexpr int bTileSMEMSize = (WMMA_K * WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks)
        * number_of_tiles_loaded_in_one_cycle;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = N;

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += 32) {
        // Load matrix A into shared memory, each thread loads 2 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 2) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[warpId * 64 + iter * 32 + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
        }

        // Load matrix B data into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const UIN bRowId = kIter + laneId;
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 512 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId * ldb + bColId] : static_cast<MATRIX_B_TYPE>(0);
        }

        __syncthreads();

        // Compute the matrix multiplication
        for (int iter = 0; iter < 32; iter += WMMA_K) {
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter, 32);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * 512 + iter, 32);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }

        }
    }
}

// m16n16k16
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block256_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                       const UIN N,
                                                                                       const UIN K,
                                                                                       const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                       const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                       const UIN numNonZeroRow,
                                                                                       const UIN *__restrict__ reorderedRows,
                                                                                       const UIN *__restrict__ reorderedCols,
                                                                                       const UIN *__restrict__ reorderedColOffset,
                                                                                       const UIN *__restrict__ blockRowOffsets,
                                                                                       const UIN *__restrict__ blockValues,
                                                                                       MATRIX_C_TYPE *matrixP) {
    constexpr int number_of_tiles_loaded_in_one_cycle = 32 / WMMA_K;

    constexpr int aTileSMEMSize = (WMMA_M * WMMA_K) * number_of_tiles_loaded_in_one_cycle;
    constexpr int bTileSMEMSize = (WMMA_K * WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks)
        * number_of_tiles_loaded_in_one_cycle;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += 32) {
        // Load matrix A into shared memory, each thread loads 2 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 2) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[warpId * 64 + iter * 32 + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);
        }

        // Load matrix B data into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const UIN bRowId = kIter + laneId;
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 512 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0);
        }

        __syncthreads();

        // Compute the matrix multiplication
        for (int iter = 0; iter < 32; iter += WMMA_K) {
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter, 32);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * 512 + iter, 32);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
__global__ void sddmm_gpu_dense_block_m16n16k8_block256_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                      const UIN N,
                                                                                      const UIN K,
                                                                                      const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                      const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                      const UIN numNonZeroRow,
                                                                                      const UIN *__restrict__ reorderedRows,
                                                                                      const UIN *__restrict__ denseCols,
                                                                                      const UIN *__restrict__ denseColOffset,
                                                                                      const UIN *__restrict__ blockOffsets,
                                                                                      const UIN *__restrict__ blockValues,
                                                                                      MATRIX_C_TYPE *matrixP) {
    constexpr int kStep = 32;
    constexpr int number_of_tiles_loaded_in_one_cycle = kStep / WMMA_K;

    const int aTileSMEMLd = (WMMA_K * number_of_tiles_loaded_in_one_cycle);
    const int bTileSMEMLd = (WMMA_K * number_of_tiles_loaded_in_one_cycle);

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize = (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) * bTileSMEMLd;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock = denseColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = denseColOffset[rowPanelId + 1];

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep) {
        // Load matrix A into shared memory, each thread loads 2 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + (warpId * 2) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[(warpId * 2 + iter) * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K) ? (matrixA[aRowId * K + aColId]) : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockId < numColBlocksCurrentRowPanel) {

            // Load matrix B into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll 8
            for (int iter = 0; iter < 16; ++iter) {
                const UIN bRowId = kIter + laneId;
                const UIN reorderedColIndex = startIndexOfDenseColsCurrentColBlock + iter;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel ?
                    denseCols[reorderedColIndex] : N;

                bTileSMEM[(warpId * WMMA_N + iter) * bTileSMEMLd + laneId] =
                    (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * K] : static_cast<MATRIX_B_TYPE>(0.0f);
            }

            // Compute the matrix multiplication
#pragma unroll
            for (int iter = 0; iter < 32; iter += WMMA_K) {
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter, aTileSMEMLd);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd + iter, bTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i) aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i) bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

__device__ void m16n16k8_block128_double_buffer_load_matrixA(const UIN matrixLd,
                                                             const UIN kIter,
                                                             const MATRIX_A_TYPE *__restrict__ matrixA,
                                                             const UIN endIndex,
                                                             const UIN rowPanelId,
                                                             const UIN *__restrict__ reorderedRows,
                                                             const int writeStage,
                                                             const int smemLd,
                                                             MATRIX_A_TYPE *aTileSMEM) {

    if (kIter >= matrixLd) {
        return;
    }

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN localRow_smem = warpId * 4 + (laneId >> 3); // shared memory location. laneId / 8
    const UIN localCol_smem = writeStage * WMMA_K + (laneId & 7); // shared memory location. laneId % 8

    const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + localRow_smem;
    const UIN localCol_gmem = kIter + (laneId & 7); // laneId % 8

    aTileSMEM[localRow_smem * smemLd + localCol_smem] = (reorderedRowIndex < endIndex && localCol_gmem < matrixLd) ?
        matrixA[reorderedRows[reorderedRowIndex] * matrixLd + localCol_gmem] : static_cast<MATRIX_A_TYPE>(0);
}

__device__ void m16n16k8_block128_double_buffer_load_matrixB(const UIN matrixLd,
                                                             const UIN kIter,
                                                             const MATRIX_B_TYPE *__restrict__ __align__(
                                                                 16) matrixB,
                                                             const UIN startIndex,
                                                             const UIN endIndex,
                                                             const UIN *__restrict__ reorderedCols,
                                                             const int smemLd,
                                                             MATRIX_B_TYPE
                                                             *bTileSMEM) {
    if (kIter >= matrixLd) {
        return;
    }

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN localRow = warpId * WMMA_N + (laneId >> 1); // shared memory location. laneId / 2
    const UIN localCol = (laneId & 1) * 4; // shared memory location. (laneId % 2) * 4

    const UIN reorderedColIndex = startIndex + localRow;
    const float4 bData = (reorderedColIndex < endIndex && kIter + localCol < matrixLd) ?
        *(float4 *) &(matrixB)[reorderedCols[reorderedColIndex] * matrixLd + kIter + localCol] :
        make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    *(float4 *) &bTileSMEM[
        localRow * smemLd
            + localCol] =
        bData;
}

// m16n16k8
// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [128,1,1]
// https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html#
__global__ void sddmm_gpu_dense_block_m16n16k8_block128_double_buffer(const UIN M,
                                                                      const UIN N,
                                                                      const UIN K,
                                                                      const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                      const MATRIX_B_TYPE *__restrict__ __align__(
                                                                          16) matrixB,
                                                                      const UIN numNonZeroRow,
                                                                      const UIN *__restrict__ reorderedRows,
                                                                      const UIN *__restrict__ reorderedCols,
                                                                      const UIN *__restrict__ reorderedColOffset,
                                                                      const UIN *__restrict__ blockRowOffsets,
                                                                      const UIN *__restrict__ blockValues,
                                                                      MATRIX_C_TYPE
                                                                      *matrixP) {
    constexpr UIN aTileSMEMLd = (WMMA_K * 2 + 4); // Double buffer and 4 padding
    constexpr UIN bTileSMEMLd = (WMMA_K + 4); // 4 padding

    constexpr UIN aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr UIN bTileSMEMSize = (WMMA_N * 4) * bTileSMEMLd;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * 4;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ __align__(16)
    MATRIX_B_TYPE bTileSMEM[bTileSMEMSize]; // col major

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag,
                  0.0f);

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentThreadBlock =
        reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

// Load first buffer
    m16n16k8_block128_double_buffer_load_matrixA(K,
                                                 0,
                                                 matrixA,
                                                 numNonZeroRow,
                                                 rowPanelId,
                                                 reorderedRows,
                                                 0,
                                                 aTileSMEMLd,
                                                 aTileSMEM);

    int writeStage = 1;

// Loop over K
    for (
        int kIter = 0;
        kIter < K;
        kIter += WMMA_K) {

// Load next buffer
        m16n16k8_block128_double_buffer_load_matrixA(K,
                                                     kIter
                                                         + WMMA_K,
                                                     matrixA,
                                                     numNonZeroRow,
                                                     rowPanelId,
                                                     reorderedRows,
                                                     writeStage,
                                                     aTileSMEMLd,
                                                     aTileSMEM);

// Load matrix B tile into shared memory and compute the matrix multiplication
        if (colBlockId < numColBlocksCurrentRowPanel) {
// Load matrix B tile into shared memory
            m16n16k8_block128_double_buffer_load_matrixB(K,
                                                         kIter,
                                                         matrixB,
                                                         startIndexOfReorderedColsCurrentThreadBlock,
                                                         endIndexOfReorderedColsCurrentPanel,
                                                         reorderedCols,
                                                         bTileSMEMLd,
                                                         bTileSMEM
            );

// load matrix A and B tile into fragment
            wmma::load_matrix_sync(aFrag, aTileSMEM
                + (writeStage ^ 1) * WMMA_K, aTileSMEMLd);
            wmma::load_matrix_sync(bFrag, bTileSMEM
                +
                    warpId * WMMA_N
                        * bTileSMEMLd, bTileSMEMLd);

// Convert to TF32
#pragma unroll
            for (
                int i = 0;
                i < aFrag.
                    num_elements;
                ++i)
                aFrag.x[i] =
                    wmma::__float_to_tf32(aFrag
                                              .x[i]);
#pragma unroll
            for (
                int i = 0;
                i < bFrag.
                    num_elements;
                ++i)
                bFrag.x[i] =
                    wmma::__float_to_tf32(bFrag
                                              .x[i]);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag
            );
        }

        __syncthreads();

        writeStage ^= 1;
    }

// Store the result
    if (colBlockId < numColBlocksCurrentRowPanel) {
#pragma unroll
        for (
            int idxOfFragment = 0;
            idxOfFragment < cFrag.
                num_elements;
            ++idxOfFragment) {
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol
            );

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

// Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
__global__ void sddmm_gpu_dense_block_m16n16k8_block256_noSMEM_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                             const UIN N,
                                                                                             const UIN K,
                                                                                             const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                             const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                             const UIN numNonZeroRow,
                                                                                             const UIN *__restrict__ reorderedRows,
                                                                                             const UIN *__restrict__ reorderedCols,
                                                                                             const UIN *__restrict__ reorderedColOffset,
                                                                                             const UIN *__restrict__ blockRowOffsets,
                                                                                             const UIN *__restrict__ blockValues,
                                                                                             MATRIX_C_TYPE *matrixP) {

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockId = blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks + warpId;
    if (colBlockId >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K, one iteration 8
    for (int kIter = 0; kIter < K; kIter += 8) {

        // Load matrix A
#pragma unroll
        for (int indexOfFragment = 0; indexOfFragment < aFrag.num_elements; ++indexOfFragment) {
            UIN localRow, localCol;
            calculateMatrixAFragmentCoordinates(laneId, indexOfFragment, localRow, localCol);

            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + localRow;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + localCol;

            aFrag.x[indexOfFragment] = (aRowId < M && aColId < K) ?
                (matrixA[aRowId * lda + aColId]) : static_cast<MATRIX_A_TYPE>(0.0f);

            if (rowPanelId == 0 && colBlockId == 0) {
                printf(
                    "colBlockId = %d, warpId = %d, laneId = %d, index = %d, localRow = %d, localCol = %d, aRowId = %d, aColId = %d, aFrag.x = %f\n",
                    colBlockId,
                    warpId,
                    laneId,
                    indexOfFragment,
                    localRow,
                    localCol,
                    aRowId,
                    aColId,
                    aFrag.x[indexOfFragment]);
            }
        }

        // Load matrix B
#pragma unroll
        for (int indexOfFragment = 0; indexOfFragment < bFrag.num_elements; ++indexOfFragment) {
            UIN localRow, localCol;
            calculateMatrixBFragmentCoordinates(laneId, indexOfFragment, localRow, localCol);

            const UIN bRowId = kIter + localRow;
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentColBlock + localCol;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bFrag.x[indexOfFragment] = (bRowId < K && bColId < N) ?
                matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0.0f);

            if (rowPanelId == 0 && colBlockId == 0) {
                printf(
                    "colBlockId = %d, warpId = %d, laneId = %d, index = %d, localRow = %d, localCol = %d, bRowId = %d, bColId = %d, bFrag.x = %f\n",
                    colBlockId,
                    warpId,
                    laneId,
                    indexOfFragment,
                    localRow,
                    localCol,
                    bRowId,
                    bColId,
                    bFrag.x[indexOfFragment]);
            }
        }

        // Convert to TF32
#pragma unroll
        for (int i = 0; i < aFrag.num_elements; ++i)aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
        for (int i = 0; i < bFrag.num_elements; ++i)bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

        __syncthreads();

        // Compute the matrix multiplication
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        __syncthreads();
    }

    // Store the result
#pragma unroll
    for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements; ++idxOfFragment) {

//        if(warpId ==0 && rowPanelId == 0){
//            printf("laneId = %d, idxOfFragment = %d, c = %f\n", laneId, idxOfFragment, c);
//        }

        UIN localRow, localCol;
        calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

        const UIN idxOfMatrixP =
            blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

        // Saved when the value is not 0
        if (idxOfMatrixP != NULL_VALUE) {
            matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
        }

        if (idxOfMatrixP == 0) {
            printf("idxOfMatrixP = %d, c = %f, blockIndex = %d \n",
                   idxOfMatrixP,
                   cFrag.x[idxOfFragment],
                   startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol);
        }
    }
}

// m16n16k8
// blockDim: [512, 1, 1]
// 一个thread block负责一个row panel中的16个col block
__global__ void sddmm_gpu_dense_block_m16n16k16_block512_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                       const UIN N,
                                                                                       const UIN K,
                                                                                       const MATRIX_A_TYPE *__restrict__ matrixA,
                                                                                       const MATRIX_B_TYPE *__restrict__ matrixB,
                                                                                       const UIN numNonZeroRow,
                                                                                       const UIN *__restrict__ reorderedRows,
                                                                                       const UIN *__restrict__ reorderedCols,
                                                                                       const UIN *__restrict__ reorderedColOffset,
                                                                                       const UIN *__restrict__ blockRowOffsets,
                                                                                       const UIN *__restrict__ blockValues,
                                                                                       MATRIX_C_TYPE *matrixP) {
    constexpr int eachThreadBlockCountsTheNumberOfColBlocks = 16;

    constexpr int aTileSMEMSize = (WMMA_M * WMMA_N) * 2;
    constexpr int bTileSMEMSize = (WMMA_K * WMMA_N * eachThreadBlockCountsTheNumberOfColBlocks) * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel = blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * eachThreadBlockCountsTheNumberOfColBlocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel) {
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock = reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel = reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K * 2) {
        // Load matrix A into shared memory, each thread loads 1 element, conflict-free access
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + warpId;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
        const UIN aColId = kIter + laneId;

        aTileSMEM[warpId * 32 + laneId] =
            (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<MATRIX_A_TYPE>(0);


        // Load matrix B data into shared memory, each thread loads 16 elements, conflict-free access
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const UIN bRowId = kIter + laneId;
            const UIN reorderedColIndex = startIndexOfReorderedColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel ?
                reorderedCols[reorderedColIndex] : N;

            bTileSMEM[warpId * 512 + iter * 32 + laneId] =
                (bRowId < K && bColId < N) ? matrixB[bRowId + bColId * ldb] : static_cast<MATRIX_B_TYPE>(0);
        }
        __syncthreads();

        // Compute the matrix multiplication
        for (int iter = 0; iter < 2; ++iter) {
            if (colBlockId < numColBlocksCurrentRowPanel) {
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter * WMMA_K, WMMA_K * 2);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * 512 + iter * WMMA_K, WMMA_K * 2);
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
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow, localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock + localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE) {
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// blockDim: [256,1,1]
__global__ void sddmm_gpu_sparse_residue_block256_rowPanel_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                         const UIN N,
                                                                                         const UIN K,
                                                                                         const float *__restrict__ matrixA,
                                                                                         const float *__restrict__ matrixB,
                                                                                         const UIN numNonZeroRow,
                                                                                         const UIN *__restrict__ reorderedRows,
                                                                                         const UIN *__restrict__ sparsePartDataOffsets,
                                                                                         const UIN *__restrict__ sparsePartData,
                                                                                         const UIN *__restrict__ relativeRows,
                                                                                         const UIN *__restrict__ sparsePartColIndices,
                                                                                         float *matrixP) {
    // 线程块中线程数量
    constexpr int numWarpsPerBlock = 8;

    constexpr int kStep = 32;

    constexpr int aTileSMEMSize = WMMA_M * kStep; // 512

    constexpr int eachThreadLoadsTheNumberOfMatrixADatas = aTileSMEMSize / (WARP_SIZE * numWarpsPerBlock); // 2
    constexpr int eachWarpLoadsTheNumberOfMatrixADatas = WARP_SIZE * eachThreadLoadsTheNumberOfMatrixADatas; // 64
    constexpr int eachWarpLoadsTheNumberOfMatrixARows = WMMA_M / numWarpsPerBlock; // 2

    __shared__ float aTileSMEM[aTileSMEMSize];

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN tId = threadIdx.x;

    const UIN rowPanelId = blockIdx.x;

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K, one iteration 32 elements
    for (int kIter = 0; kIter < K; kIter += kStep) {
        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (int rowIter = 0; rowIter < eachWarpLoadsTheNumberOfMatrixARows; ++rowIter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) +
                (warpId * eachWarpLoadsTheNumberOfMatrixARows) + rowIter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
#pragma unroll
            for (int colIter = 0; colIter < kStep; colIter += WARP_SIZE) {
                const UIN aColId = kIter + colIter + laneId;

                aTileSMEM[warpId * eachWarpLoadsTheNumberOfMatrixADatas + rowIter * kStep + colIter + laneId] =
                    (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<float>(0);
            }
        }

        __syncthreads();

        // Load matrix B and compute the matrix multiplication
        for (int iter = sparsePartDataOffsets[rowPanelId] + tId;
             iter < sparsePartDataOffsets[rowPanelId + 1];
             iter += blockDim.x) { // Iterate over all the sparse data in the current row panel
            const UIN relativeRow = relativeRows[iter];
            const UIN col = sparsePartColIndices[iter];
            const UIN indexOfMatrixP = sparsePartData[iter];

            float c = 0.0f;
            for (int localKIter = 0; localKIter < kStep; localKIter += 4) {
                const float4 aData = *((float4 *) &aTileSMEM[relativeRow * kStep + localKIter]);
                const float4 bData = *((float4 *) &matrixB[col * ldb + kIter + localKIter]);
                c += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z + aData.w * bData.w;
            }

            matrixP[indexOfMatrixP] += c;
        }

        __syncthreads();
    }
}

// blockDim: [256,1,1]
__global__ void sddmm_gpu_sparse_residue_block256_matrixA_rowMaj_matrixB_colMaj(const UIN M,
                                                                                const UIN N,
                                                                                const UIN K,
                                                                                const float *__restrict__ matrixA,
                                                                                const float *__restrict__ matrixB,
                                                                                const UIN numNonZeroRow,
                                                                                const UIN *__restrict__ reorderedRows,
                                                                                const UIN *__restrict__ sparsePartDataOffsets,
                                                                                const UIN *__restrict__ sparsePartData,
                                                                                const UIN *__restrict__ relativeRows,
                                                                                const UIN *__restrict__ sparsePartColIndices,
                                                                                float *matrixP) {
    // 线程块中线程数量
    constexpr int numWarpsPerBlock = 8;
    constexpr int numThreadsPerBlock = numWarpsPerBlock * WARP_SIZE; // 256

    constexpr int kStep = 32;

    constexpr int aTileSMEMSize = WMMA_M * kStep; // 512
    constexpr int cSMEMSize = numThreadsPerBlock; // 256

    constexpr int eachThreadLoadsTheNumberOfMatrixADatas = aTileSMEMSize / (WARP_SIZE * numWarpsPerBlock); // 2
    constexpr int eachWarpLoadsTheNumberOfMatrixADatas = WARP_SIZE * eachThreadLoadsTheNumberOfMatrixADatas; // 64
    constexpr int eachWarpLoadsTheNumberOfMatrixARows = WMMA_M / numWarpsPerBlock; // 2

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock = sparsePartDataOffsets[rowPanelId] + blockIdx.y * cSMEMSize;
    const UIN indexBoundaryCurrentRowPanel = sparsePartDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel) {
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + threadIdx.x;

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparsePartColIndices[index];

    __shared__ float aTileSMEM[aTileSMEMSize];
    __shared__ float pSMEM[cSMEMSize];

    pSMEM[threadIdx.x] = 0.0f;

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K, one iteration 32 elements
    for (int kIter = 0; kIter < K; kIter += kStep) {
        // Load matrix A into shared memory, conflict-free access
#pragma unroll 2
        for (int rowIter = 0; rowIter < eachWarpLoadsTheNumberOfMatrixARows; ++rowIter) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) +
                (warpId * eachWarpLoadsTheNumberOfMatrixARows) + rowIter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[warpId * eachWarpLoadsTheNumberOfMatrixADatas + rowIter * kStep + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * lda + aColId] : static_cast<float>(0);
        }

        __syncthreads();

        // Load matrix B and compute the matrix multiplication
        if (index < indexBoundaryCurrentRowPanel) {
#pragma unroll 4
            for (int localKIter = 0; localKIter < kStep; localKIter += 4) {
                const float4 aData = *((float4 *) &aTileSMEM[relativeRow * kStep + localKIter]);
                const float4 bData = *((float4 *) &matrixB[col * ldb + kIter + localKIter]);
                pSMEM[threadIdx.x] += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z + aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    if (index < indexBoundaryCurrentRowPanel) {
        matrixP[sparsePartData[index]] = pSMEM[threadIdx.x];
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block(const UIN M,
                                       const UIN N,
                                       const UIN K,
                                       const float *__restrict__ matrixA,
                                       const float *__restrict__ matrixB,
                                       const UIN numNonZeroRow,
                                       const UIN *__restrict__ reorderedRows,
                                       const UIN *__restrict__ sparseDataOffsets,
                                       const UIN *__restrict__ sparseData,
                                       const UIN *__restrict__ relativeRows,
                                       const UIN *__restrict__ sparseColIndices,
                                       float *matrixP) {

    constexpr int kStep = 32;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock = sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel =
        ::min(startIndexOfSparseDataCurrentBlock + calculateDataPerThreadBlock, sparseDataOffsets[rowPanelId + 1]);

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel) {
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + tId;

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseColIndices[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep) {

        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (UIN iter = warpId; iter < WMMA_M; iter += numWarps) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[iter * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * K + aColId] : static_cast<float>(0);
        }

        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate one element
        if (index < indexBoundaryCurrentRowPanel) {
#pragma unroll
            for (int localKIter = 0; localKIter < kStep; localKIter += 4) {
                const float4 aData = *((float4 *) &aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 bData = *((float4 *) &matrixB[col * K + kIter + localKIter]);
                c += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z + aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    if (index < indexBoundaryCurrentRowPanel) {
        matrixP[sparseData[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_2threadOneData_shuffle(const UIN M,
                                                              const UIN N,
                                                              const UIN K,
                                                              const float *__restrict__ matrixA,
                                                              const float *__restrict__ matrixB,
                                                              const UIN numNonZeroRow,
                                                              const UIN *__restrict__ reorderedRows,
                                                              const UIN *__restrict__ sparseValueOffsets,
                                                              const UIN *__restrict__ sparseValues,
                                                              const UIN *__restrict__ relativeRows,
                                                              const UIN *__restrict__ sparseCols,
                                                              float *matrixP) {

    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock = sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseValueOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel = sparseValueOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel) {
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + (tId >> 1);

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseCols[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段, 避免了线程之间的数据竞争
    const UIN oddOrEven = laneId & 1;

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep) {

        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (UIN iter = warpId; iter < WMMA_M; iter += numWarps) {
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[iter * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * K + aColId] : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate one element
        if (index < indexBoundaryCurrentRowPanel) {
#pragma unroll
            for (int localKIter = oddOrEven * kStepPerThread; localKIter < (oddOrEven + 1) * kStepPerThread;
                 localKIter += 4) {
                const float4 aData = *((float4 *) &aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 bData = *((float4 *) &matrixB[col * K + kIter + localKIter]);
                c += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z + aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    // Use the shuffle instruction to merge the results of two adjacent threads.
    const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
    c += __shfl_xor_sync(mask, c, 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程0的sm1上


    if (index < indexBoundaryCurrentRowPanel && oddOrEven == 0) {
        matrixP[sparseValues[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_block512_2thread_mutData_shuffle(const UIN M,
                                                                        const UIN N,
                                                                        const UIN K,
                                                                        const float *__restrict__ matrixA,
                                                                        const float *__restrict__ matrixB,
                                                                        const UIN numNonZeroRow,
                                                                        const UIN *__restrict__ reorderedRows,
                                                                        const UIN *__restrict__ sparseDataOffsets,
                                                                        const UIN *__restrict__ sparseData,
                                                                        const UIN *__restrict__ relativeRows,
                                                                        const UIN *__restrict__ sparseColIndices,
                                                                        float *matrixP) {

    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock = sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;
    constexpr int calculateDataPerThread = 2;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN endIndexCurrentRowPanel = sparseDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= endIndexCurrentRowPanel) {
        return;
    }

    const UIN startIndex = startIndexOfSparseDataCurrentBlock + (tId >> 1) * 2;

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段, 避免了线程之间的数据竞争
    const UIN oddOrEven = laneId & 1;

    float c[2] = {0.0f, 0.0f};

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep) {

        // Load matrix A into shared memory, conflict-free access
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + warpId;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
        const UIN aColId = kIter + laneId;
        aTileSMEM[warpId * aTileSMEM_ld + laneId] =
            (aRowId < M && aColId < K) ? matrixA[aRowId * K + aColId] : static_cast<float>(0);
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate one element
#pragma unroll
        for (int i = 0; i < 2 && startIndex + i < endIndexCurrentRowPanel; ++i) {
            const UIN relativeRow = relativeRows[startIndex + i];
            const UIN col = sparseColIndices[startIndex + i];
#pragma unroll
            for (int localKIter = oddOrEven * kStepPerThread; localKIter < (oddOrEven + 1) * kStepPerThread;
                 localKIter += 4) {
                const float4 aData = *((float4 *) &aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 bData = *((float4 *) &matrixB[col * K + kIter + localKIter]);
                c[i] += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z + aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < calculateDataPerThread;
         ++i) {    // Use the shuffle instruction to merge the results of two adjacent threads.
        const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
        c[i] += __shfl_xor_sync(mask, c[i], 1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上, 线程1的sm1加到线程0的sm1上
    }

    for (int i = 0; i < calculateDataPerThread && startIndex + i < endIndexCurrentRowPanel && oddOrEven == 0; ++i) {
        matrixP[sparseData[startIndex + i]] = c[i];
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_block512_warpOneData_shuffle(const UIN M,
                                                                    const UIN N,
                                                                    const UIN K,
                                                                    const float *__restrict__ matrixA,
                                                                    const float *__restrict__ matrixB,
                                                                    const UIN numNonZeroRow,
                                                                    const UIN *__restrict__ reorderedRows,
                                                                    const UIN *__restrict__ sparseDataOffsets,
                                                                    const UIN *__restrict__ sparseData,
                                                                    const UIN *__restrict__ relativeRows,
                                                                    const UIN *__restrict__ sparseColIndices,
                                                                    float *matrixP) {
    // 线程块中线程数量
    constexpr int numWarpsPerBlock = 16;
    constexpr UIN calculateDataPerThreadBlock = numWarpsPerBlock;

    constexpr int kStep = 32;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel = sparseDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel) {
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + warpId;

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseColIndices[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep) {

        // Load matrix A into shared memory, conflict-free access
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + warpId;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
        const UIN aColId = kIter + laneId;
        aTileSMEM[warpId * aTileSMEM_ld + laneId] =
            (aRowId < M && aColId < K) ? matrixA[aRowId * K + aColId] : static_cast<float>(0);
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate one element
        if (index < indexBoundaryCurrentRowPanel) {
            const float aData = aTileSMEM[relativeRow * aTileSMEM_ld + laneId];
            const float bData = matrixB[col * K + kIter + laneId];
            c += aData * bData;
        }

        __syncthreads();
    }

    c = cuUtil::warp_reduce_sum(c);

    if (index < indexBoundaryCurrentRowPanel && laneId == 0) {
        matrixP[sparseData[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_warpMutData_shuffle(const UIN M,
                                                           const UIN N,
                                                           const UIN K,
                                                           const float *__restrict__ matrixA,
                                                           const float *__restrict__ matrixB,
                                                           const UIN numNonZeroRow,
                                                           const UIN *__restrict__ reorderedRows,
                                                           const UIN *__restrict__ sparseDataOffsets,
                                                           const UIN *__restrict__ sparseData,
                                                           const UIN *__restrict__ relativeRows,
                                                           const UIN *__restrict__ sparseColIndices,
                                                           float *matrixP) {
    constexpr int kStep = 32;

    constexpr int aTileSMEM_ld = kStep;
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN numWarps = (blockDim.x + 31) >> 5;

    const UIN calculateDataPerWarp = sddmm_sparse_block_each_thread_block_counts_the_number_Of_data / numWarps;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] + blockIdx.y * sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;
    const UIN indexBoundaryCurrentRowPanel = sparseDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel) {
        return;
    }

    __shared__ float aTileSMEM[aTileSMEMSize];
    extern __shared__ float pSMEM[]; // sddmm_sparse_block_each_thread_block_counts_the_number_Of_data

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep) {

        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (int rowIter = 0; rowIter < WMMA_M; rowIter += numWarps) {
            const UIN smemRowId = rowIter + warpId;
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRowId;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow ? reorderedRows[reorderedRowIndex] : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[smemRowId * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K) ? matrixA[aRowId * K + aColId] : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication
#pragma unroll
        for (int iter = 0; iter < calculateDataPerWarp; ++iter) {
            const UIN index = startIndexOfSparseDataCurrentBlock + warpId * calculateDataPerWarp + iter;
            const UIN relativeRow = relativeRows[index];
            const UIN col = sparseColIndices[index];
            const float aData = aTileSMEM[relativeRow * aTileSMEM_ld + laneId];
            const float bData = matrixB[col * K + kIter + laneId];
            float c = aData * bData;

            c = cuUtil::warp_reduce_sum(c);
            if (laneId == 0) {
                pSMEM[warpId * calculateDataPerWarp + iter] += c;
            }
        }

        __syncthreads();
    }

    if (tId < sddmm_sparse_block_each_thread_block_counts_the_number_Of_data) {
        const UIN index = startIndexOfSparseDataCurrentBlock + tId;
        matrixP[sparseData[index]] = pSMEM[tId];
    }
}

} // namespace kernel

void sddmm_gpu(const Matrix<float> &matrixA,
               const Matrix<float> &matrixB,
               const sparseMatrix::CSR<float> &matrixS,
               const ReBELL &rebell,
               sparseMatrix::CSR<float> &matrixP,
               float &time) {

    dev::vector<float> matrixA_dev(matrixA.values());
    dev::vector<float> matrixB_dev(matrixB.values());
    dev::vector<float> matrixP_dev(matrixS.values().size(), 0);

    sddmm_gpu(matrixS.row(),
              matrixS.col(),
              matrixA.col(),
              matrixA_dev.data(),
              matrixB_dev.data(),
              rebell,
              matrixP_dev.data(),
              time);

    // Copy the results from the device to the host
    matrixP.setValues() = d2h(matrixP_dev);
}

void sddmm_gpu(UIN M, UIN N, UIN K,
               const float *matrixA,
               const float *matrixB,
               const ReBELL &rebell,
               float *matrixP,
               float &time) {

//    // Convert the data type of matrix A and matrix B for use tensor core
//    dev::vector<MATRIX_A_TYPE> matrixA_convertedType(M * K);
//    dev::vector<MATRIX_B_TYPE> matrixB_convertedType(N * K);
//    {
//        const int numThreadPerBlock = 1024;
//        kernel::convertDataType<<< (M * K + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
//            M * K, matrixA, matrixA_convertedType.data());
//        kernel::convertDataType<<< (N * K + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
//            N * K, matrixB, matrixB_convertedType.data());
//    }

    dim3 grid_dense, block_dense, grid_sparse, block_sparse;

    block_dense.x = WARP_SIZE * sddmm_dense_block_number_of_warps_per_thread_block;
    // Assign row panel to x-axis of grid, and assign col block to y-axis of grid
    grid_dense.x = rebell.numRowPanels();
    grid_dense.y = std::ceil(static_cast<float>(rebell.maxNumDenseColBlocks())
                                 / each_thread_block_counts_the_number_Of_dense_blocks);

    block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
    grid_sparse.x = rebell.numRowPanels();
    grid_sparse.y = rebell.maxNumSparseColBlocks();

    printf("grid_dense: [%u, %u, %u], block_dense: [%u, %u, %u]\n",
           grid_dense.x, grid_dense.y, grid_dense.z,
           block_dense.x, block_dense.y, block_dense.z);
    printf("grid_sparse: [%u, %u, %u], block_sparse: [%u, %u, %u]\n",
           grid_sparse.x, grid_sparse.y, grid_sparse.z,
           block_sparse.x, block_sparse.y, block_sparse.z);

    cudaStream_t denseStream;
    cudaStream_t sparseStream;

    cudaStreamCreate(&denseStream);
    cudaStreamCreate(&sparseStream);

    CudaTimeCalculator totalTimeCalculator, denseKernelTimeCalculator, sparseKernelTimeCalculator;

    totalTimeCalculator.startClock();

    denseKernelTimeCalculator.startClock(denseStream);

#ifdef WMMA_16_16_16
    kernel::sddmm_gpu_dense_block_m16n16k16_block256_matrixA_rowMaj_matrixB_colMaj<<<grid_rebell, block_rebell>>>(matrixS.row(), matrixS.col(), matrixA.col(),
        matrixA_convertedType.data(),
        matrixB_convertedType.data(),
        rebell.reorderedRows().size(),
        reorderedRowIndices_dev.data(),
        reorderedColIndices_dev.data(),
        reorderedColIndicesOffset_dev.data(),
        blockRowOffsets_dev.data(),
        blockValues_dev.data(),
        matrixP_dev.data());
#endif // WMMA_16_16_16

#ifdef WMMA_16_16_8
    kernel::sddmm_gpu_dense_block_m16n16k8_block256_matrixA_rowMaj_matrixB_colMaj<<<grid_dense, block_dense, 0, denseStream>>>(M, N, K,
        matrixA,
        matrixB,
        rebell.reorderedRows().size(),
        rebell.reorderedRows().data(),
        rebell.denseCols().data(),
        rebell.denseColOffsets().data(),
        rebell.blockOffsets().data(),
        rebell.blockValues().data(),
        matrixP);
#endif // WMMA_16_16_8

    denseKernelTimeCalculator.endClock(denseStream);

    sparseKernelTimeCalculator.startClock(sparseStream);

    kernel::sddmm_gpu_sparse_block_2threadOneData_shuffle<<<grid_sparse, block_sparse, 0, sparseStream>>>(M, N, K,
        matrixA,
        matrixB,
        rebell.reorderedRows().size(),
        rebell.reorderedRows().data(),
        rebell.sparseValueOffsets().data(),
        rebell.sparseValues().data(),
        rebell.sparseRelativeRows().data(),
        rebell.sparseColIndices().data(),
        matrixP);

    sparseKernelTimeCalculator.endClock(sparseStream);

    totalTimeCalculator.endClock();

    const float denseBlockTime = denseKernelTimeCalculator.getTime();
    const float sparseBlockTime = sparseKernelTimeCalculator.getTime();
    const float totalTime = totalTimeCalculator.getTime();

    const float overlapEfficiency = (denseBlockTime + sparseBlockTime) / totalTime;

    printf("denseBlockTime: %f ms\n", denseBlockTime);
    printf("sparseBlockTime: %f ms\n", sparseBlockTime);
    printf("totalTime: %f ms, overlapEfficiency: %f\n", totalTime, overlapEfficiency);

    time = totalTime;
}

void sddmm_gpu_batch(UIN numBatch, UIN M, UIN N, UIN K,
                     const float *matrixA,
                     const float *matrixB,
                     const std::vector<ReBELL> &rebell,
                     float *matrixP,
                     float &time) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int batchId = 0; batchId < numBatch; ++batchId) {

        dim3 grid_dense, block_dense, grid_sparse, block_sparse;

        block_dense.x = WARP_SIZE * sddmm_dense_block_number_of_warps_per_thread_block;
        // Assign row panel to x-axis of grid, and assign col block to y-axis of grid
        grid_dense.x = rebell[batchId].numRowPanels();
        grid_dense.y = std::ceil(static_cast<float>(rebell[batchId].maxNumDenseColBlocks())
                                     / each_thread_block_counts_the_number_Of_dense_blocks);

        block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
        grid_sparse.x = rebell[batchId].numRowPanels();
        grid_sparse.y = rebell[batchId].maxNumSparseColBlocks();

        kernel::sddmm_gpu_dense_block_m16n16k8_block256_matrixA_rowMaj_matrixB_colMaj<<<grid_dense, block_dense, 0, stream>>>(M, N, K,
            matrixA,
            matrixB,
            rebell[batchId].reorderedRows().size(),
            rebell[batchId].reorderedRows().data(),
            rebell[batchId].denseCols().data(),
            rebell[batchId].denseColOffsets().data(),
            rebell[batchId].blockOffsets().data(),
            rebell[batchId].blockValues().data(),
            matrixP);

        kernel::sddmm_gpu_sparse_block_2threadOneData_shuffle<<<grid_sparse, block_sparse, 0, stream>>>(M, N, K,
            matrixA,
            matrixB,
            rebell[batchId].reorderedRows().size(),
            rebell[batchId].reorderedRows().data(),
            rebell[batchId].sparseValueOffsets().data(),
            rebell[batchId].sparseValues().data(),
            rebell[batchId].sparseRelativeRows().data(),
            rebell[batchId].sparseColIndices().data(),
            matrixP);
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    CudaTimeCalculator cudaTimeCalculator;
    cudaTimeCalculator.startClock(stream);

    cudaGraphLaunch(graphExec, stream);

    cudaTimeCalculator.endClock(stream);
    cudaStreamSynchronize(stream);

    const float totalTime = cudaTimeCalculator.getTime();
    printf("sddmm_gpu_batch: numBatch = %d, totalTime: %f ms\n", numBatch, totalTime);
    time = totalTime;

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}