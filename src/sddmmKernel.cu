#include <mma.h>

#include <cstdio>

#include "BSMR.hpp"
#include "CudaTimeCalculator.cuh"
#include "Logger.hpp"
#include "TensorCoreConfig.cuh"
#include "cudaUtil.cuh"
#include "sddmmKernel.cuh"

#include <thrust/system/cuda/detail/core/util.h>

namespace kernel{
using namespace nvcuda;

__global__ void checkFragmentData(){
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

    if (warpId == 0 && laneId == 0){
        for (int i = 0; i < aTileSize; ++i){
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
        if (bRow == wmmaK){
            for (int i = 0; i < bTileSize; ++i){
                bTileSMEM[i] = static_cast<matrixBType>(i);
            }
        }
        else{
            for (int row = 0; row < wmmaK; ++row){
                for (int col = 0; col < wmmaN; ++col){
                    bTileSMEM[row + col * ldBTile] =
                        static_cast<matrixBType>(row * wmmaN + col);
                }
            }
        }
    }

    if (warpId == 0 && laneId == 0){
        printf("\nmatrix A data : \n");
        printf("| |");
        for (int col = 0; col < wmmaK; ++col){
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < wmmaK + 1; ++i){
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < wmmaM; ++row){
            printf("|%d|", row);
            for (int col = 0; col < wmmaK; ++col){
                printf("%.0f|", static_cast<float>(aTileSMEM[row * wmmaK + col]));
            }
            printf("\n");
        }

        printf("\nmatrix B data : ");
        if (ldBTile == wmmaN){
            printf("(rwo major)\n");
        }
        else{
            printf("(column major)\n");
        }
        printf("| |");
        for (int col = 0; col < bCol; ++col){
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < bCol + 1; ++i){
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < bRow; ++row){
            printf("|%d|", row);
            for (int col = 0; col < bCol; ++col){
                printf("%.0f|", static_cast<float>(bTileSMEM[row * ldBTile + col]));
            }
            printf("\n");
        }
        printf("\n");

        printf("\nmatrix C data : \n");
        printf("| |");
        for (int col = 0; col < wmmaN; ++col){
            printf("%d|", col);
        }
        printf("\n");

        printf("|");
        for (int i = 0; i < wmmaN + 1; ++i){
            printf("-|");
        }
        printf("\n");

        for (int row = 0; row < wmmaM; ++row){
            printf("|%d|", row);
            for (int col = 0; col < wmmaN; ++col){
                float c = 0.0f;
                for (int k = 0; k < wmmaK; ++k){
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

    if (warpId == 0){
        wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, matrixATypeFragment,
                       wmma::row_major>
            aFrag;
        wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, matrixBTypeFragment,
                       wmma::col_major>
            bFrag;

        wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> cFrag;

        fill_fragment(cFrag, 0.0f);

        wmma::load_matrix_sync(aFrag, aTileSMEM, ldATile);
        wmma::load_matrix_sync(bFrag, bTileSMEM, ldBTile);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

        if (laneId == 0){
            printf("\nFragment A tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx){
            if (warpId == 0 && laneId == laneIdx){
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < aFrag.num_elements;
                     ++idxOfFragment){
                    printf("%.0f|", static_cast<float>(aFrag.x[idxOfFragment]));
                }
                printf("\n");
            }
        }

        if (laneId == 0){
            printf("\nFragment B tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx){
            if (warpId == 0 && laneId == laneIdx){
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < bFrag.num_elements;
                     ++idxOfFragment){
                    printf("%.0f|", static_cast<float>(bFrag.x[idxOfFragment]));
                }
                printf("\n");
            }
        }

        if (laneId == 0){
            printf("\nFragment C tiled data : \n");
        }
        for (int laneIdx = 0; laneIdx < WARP_SIZE; ++laneIdx){
            if (warpId == 0 && laneId == laneIdx){
                printf("|T%d|", laneId);
                for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements;
                     ++idxOfFragment){
                    printf("%.0f|", static_cast<float>(cFrag.x[idxOfFragment]));
                }
                printf("\n");
            }
        }
    }
}

// m16n16k8
// 一个warp负责row panel中的1个col block
__global__ void sddmm_gpu_dense_block_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = kStep + 4;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId]);
    const UIN endBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId + 1]);
    const UIN numColBlocksCurrentRowPanel = endBlockIdCurrentRowPanel - startBlockIdCurrentRowPanel;

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockIdCurrentRowPanel = colBlockIter + warpId;
    const UIN colBlockId = startBlockIdCurrentRowPanel + colBlockIdCurrentRowPanel;
    const UIN startIndexOfBlockValuesCurrentBlock = colBlockId * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock = BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = endBlockIdCurrentRowPanel * BLOCK_COL_SIZE;

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory
#pragma unroll
        for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[smemRow * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K)
                    ? __ldg(&matrixA[aRowId * K + aColId])
                    : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockIdCurrentRowPanel < numColBlocksCurrentRowPanel){
            // Load matrix B into shared memory
#pragma unroll 8
            for (int iter = 0; iter < 16; ++iter){
                const UIN bRowId = kIter + laneId;
                const UIN reorderedColIndex =
                    startIndexOfDenseColsCurrentColBlock + iter;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                       ? denseCols[reorderedColIndex]
                                       : N;

                bTileSMEM[(warpId * WMMA_N + iter) * bTileSMEMLd + laneId] =
                    (bRowId < K && bColId < N)
                        ? __ldg(&matrixB[bRowId + bColId * K])
                        : static_cast<MATRIX_B_TYPE>(0.0f);
            }

            // Compute the matrix multiplication
#pragma unroll
            for (int localK = 0; localK < 32; localK += WMMA_K){
                wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
                wmma::load_matrix_sync(
                    bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd + localK,
                    bTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i)
                    aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i)
                    bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

                wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockIdCurrentRowPanel < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// 一个warp负责row panel中的1个col block
__global__ void sddmm_gpu_dense_block_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = kStep + 4;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    // __shared__ UIN startBlockIdCurrentRowPanel;
    // __shared__ UIN endBlockIdCurrentRowPanel;
    // if (threadIdx.x == 0){
    //     startBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId]);
    //     endBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId + 1]);
    // }
    // __syncthreads();
    const UIN startBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId]);
    const UIN endBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId + 1]);

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= endBlockIdCurrentRowPanel - startBlockIdCurrentRowPanel){
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockId = startBlockIdCurrentRowPanel + colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = colBlockId * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock = BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = endBlockIdCurrentRowPanel * BLOCK_COL_SIZE;

#pragma unroll
    for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? __ldg(&reorderedRows[reorderedRowIndex])
                               : M;
        const UIN aColId = laneId;

        aTileSMEM[smemRow * aTileSMEMLd + laneId] =
            (aRowId < M && aColId < K)
                ? __ldg(&matrixA[aRowId * K + aColId])
                : static_cast<MATRIX_A_TYPE>(0.0f);
    }

    __syncthreads();

    if (colBlockId < endBlockIdCurrentRowPanel){
#pragma unroll
        for (int iter = 0; iter < 16; ++iter){
            const UIN bRowId = laneId;
            const UIN reorderedColIndex =
                startIndexOfDenseColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                   ? denseCols[reorderedColIndex]
                                   : N;

            bTileSMEM[(warpId * WMMA_N + iter) * bTileSMEMLd + laneId] =
                (bRowId < K && bColId < N)
                    ? __ldg(&matrixB[bRowId + bColId * K])
                    : static_cast<MATRIX_B_TYPE>(0.0f);
        }

        // Compute the matrix multiplication
#pragma unroll
        for (int localK = 0; localK < 32; localK += WMMA_K){
            wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
            wmma::load_matrix_sync(
                bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd + localK,
                bTileSMEMLd);

            // Convert to TF32
#pragma unroll
            for (int i = 0; i < aFrag.num_elements; ++i)
                aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
            for (int i = 0; i < bFrag.num_elements; ++i)
                bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }


    // Store the result
    if (colBlockId < endBlockIdCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                __ldg(&blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol]);

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

__global__ void sddmm_gpu_dense_block_k32_lianxu_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ denseColOffset,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = WMMA_K;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        __ldg(&blockOffsets[rowPanelId + 1]) - __ldg(&blockOffsets[rowPanelId]);

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (__ldg(&blockOffsets[rowPanelId]) + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock =
        __ldg(&denseColOffset[rowPanelId]) + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = __ldg(&denseColOffset[rowPanelId + 1]);

#pragma unroll
    for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? __ldg(&reorderedRows[reorderedRowIndex])
                               : M;
        const UIN aColId = laneId;

        aTileSMEM[smemRow * aTileSMEMLd + laneId] =
            (aRowId < M && aColId < K)
                ? __ldg(&matrixA[aRowId * K + aColId])
                : static_cast<MATRIX_A_TYPE>(0.0f);
    }

    __syncthreads();

    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int iter = 0; iter < 4; ++iter){
            const UIN localK = iter * WMMA_K;
            const UIN bRowId = localK + (laneId % 2) * 4;
            const UIN reorderedColIndex =
                startIndexOfDenseColsCurrentColBlock + laneId / 2;
            const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                   ? __ldg(&denseCols[reorderedColIndex])
                                   : N;
            const float4 bData = (bRowId < K && bColId < N)
                                     ? __ldg(reinterpret_cast<const float4*>(
                                         &matrixB[bColId * K + bRowId]))
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const UIN smemOffset =
                (warpId * WMMA_N + laneId / 2) * bTileSMEMLd + (laneId % 2) * 4;
            *((float4*)(&bTileSMEM[smemOffset])) = bData;

            wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd,
                                   bTileSMEMLd);

            // Convert to TF32
#pragma unroll
            for (int i = 0; i < aFrag.num_elements; ++i)
                aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
            for (int i = 0; i < bFrag.num_elements; ++i)
                bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }


    // Store the result
    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                __ldg(&blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol]);

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// 一个warp负责row panel中的1个col block
// 解决 Load imbalance的问题
__global__ void sddmm_gpu_dense_block_2_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    const UIN* __restrict__ rowPanelIds,
    const UIN* __restrict__ colBlockIters,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = kStep + 4;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = rowPanelIds[blockIdx.x];
    const UIN colBlockIterCurrentThreadBlock = colBlockIters[blockIdx.x];
    // UIN rowPanelId = 0;
    // UIN colBlockIterCurrentThreadBlock = 0;
    // {
    //     UIN numBlocks = 0;
    //
    //     while (rowPanelId * ROW_PANEL_SIZE < numNonZeroRow){
    //         numBlocks += (blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId] +
    //                 each_thread_block_counts_the_number_Of_cols - 1) /
    //             each_thread_block_counts_the_number_Of_cols;
    //         if (numBlocks > blockIdx.x){
    //             break;
    //         }
    //         ++rowPanelId;
    //     }
    //     colBlockIterCurrentThreadBlock = blockIdx.x - numBlocks;
    // }

    const UIN endBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId + 1]);

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockId = colBlockIterCurrentThreadBlock + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = colBlockId * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock = BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = endBlockIdCurrentRowPanel * BLOCK_COL_SIZE;

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, each thread loads 2 elements,
        // conflict-free access
#pragma unroll
        for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[smemRow * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K)
                    ? __ldg(&matrixA[aRowId * K + aColId])
                    : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockId < endBlockIdCurrentRowPanel){
            // Load matrix B into shared memory, each thread loads 16 elements,
            // conflict-free access
#pragma unroll 8
            for (int iter = 0; iter < 16; ++iter){
                const UIN bRowId = kIter + laneId;
                const UIN reorderedColIndex =
                    startIndexOfDenseColsCurrentColBlock + iter;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                       ? denseCols[reorderedColIndex]
                                       : N;

                bTileSMEM[(warpId * WMMA_N + iter) * bTileSMEMLd + laneId] =
                    (bRowId < K && bColId < N)
                        ? __ldg(&matrixB[bRowId + bColId * K])
                        : static_cast<MATRIX_B_TYPE>(0.0f);
            }

            // Compute the matrix multiplication
#pragma unroll
            for (int localK = 0; localK < 32; localK += WMMA_K){
                wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
                wmma::load_matrix_sync(
                    bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd + localK,
                    bTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i)
                    aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i)
                    bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

                wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockId < endBlockIdCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }

            // if (idxOfMatrixP == 50528){
            //     printf(" idxOfMatrixP: %u, "
            //            "accFrag.x[%d]: %f, "
            //            "localRow: %u, "
            //            "localCol: %u, "
            //            "colBlockIdCurrentRowPanel: %u, "
            //            "numColBlocksCurrentRowPanel: %u, "
            //            "startIndexOfBlockValuesCurrentBlock: %u, "
            //            "rowPanelId: %u, "
            //            "blockIdx.x: %u, "
            //            "warpId: %u, "
            //            "laneId: %u, "
            //            "idxOfFragment = %d, "
            //            "\n",
            //            idxOfMatrixP, idxOfFragment, accFrag.x[idxOfFragment],
            //            localRow, localCol, colBlockIdCurrentRowPanel,
            //            numColBlocksCurrentRowPanel,
            //            startIndexOfBlockValuesCurrentBlock,
            //            rowPanelId,
            //            blockIdx.x,
            //            warpId, laneId,
            //            idxOfFragment);
            // }
        }
    }
}

// m16n16k8
// 一个warp负责row panel中的1个col block
// 解决 Load imbalance的问题
__global__ void sddmm_gpu_dense_block_2_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    const UIN* __restrict__ rowPanelIds,
    const UIN* __restrict__ colBlockIters,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = WMMA_K;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = rowPanelIds[blockIdx.x];
    const UIN colBlockIterCurrentThreadBlock = colBlockIters[blockIdx.x];
    // UIN rowPanelId = 0;
    // UIN colBlockIterCurrentThreadBlock = 0;
    // {
    //     UIN numBlocks = 0;
    //
    //     while (rowPanelId * ROW_PANEL_SIZE < numNonZeroRow){
    //         numBlocks += (blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId] +
    //                 each_thread_block_counts_the_number_Of_cols - 1) /
    //             each_thread_block_counts_the_number_Of_cols;
    //         if (numBlocks > blockIdx.x){
    //             break;
    //         }
    //         ++rowPanelId;
    //     }
    //     colBlockIterCurrentThreadBlock = blockIdx.x - numBlocks;
    // }

    const UIN endBlockIdCurrentRowPanel = __ldg(&blockOffsets[rowPanelId + 1]);

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockId = colBlockIterCurrentThreadBlock + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock = colBlockId * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock = BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = endBlockIdCurrentRowPanel * BLOCK_COL_SIZE;

#pragma unroll
    for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? reorderedRows[reorderedRowIndex]
                               : M;
        const UIN aColId = laneId;

        aTileSMEM[smemRow * aTileSMEMLd + laneId] =
            (aRowId < M && aColId < K)
                ? __ldg(&matrixA[aRowId * K + aColId])
                : static_cast<MATRIX_A_TYPE>(0.0f);
    }

    __syncthreads();

    if (colBlockId < endBlockIdCurrentRowPanel){
#pragma unroll
        for (int iter = 0; iter < 4; ++iter){
            const UIN localK = iter * WMMA_K;
            const UIN bRowId = localK + (laneId % 2) * 4;
            const UIN reorderedColIndex =
                startIndexOfDenseColsCurrentColBlock + laneId / 2;
            const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                   ? denseCols[reorderedColIndex]
                                   : N;
            const float4 bData = (bRowId < K && bColId < N)
                                     ? __ldg(reinterpret_cast<const float4*>(
                                         &matrixB[bColId * K + bRowId]))
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const UIN smemOffset =
                (warpId * WMMA_N + laneId / 2) * bTileSMEMLd + (laneId % 2) * 4;
            // *((float4*)(&bTileSMEM[smemOffset])) = bData;
            bTileSMEM[smemOffset] = bData.x;
            bTileSMEM[smemOffset + 1] = bData.y;
            bTileSMEM[smemOffset + 2] = bData.z;
            bTileSMEM[smemOffset + 3] = bData.w;

            wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd,
                                   bTileSMEMLd);

            // Convert to TF32
#pragma unroll
            for (int i = 0; i < aFrag.num_elements; ++i)
                aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
            for (int i = 0; i < bFrag.num_elements; ++i)
                bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    // Store the result
    if (colBlockId < endBlockIdCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }

            // if (idxOfMatrixP == 50528){
            //     printf(" idxOfMatrixP: %u, "
            //            "accFrag.x[%d]: %f, "
            //            "localRow: %u, "
            //            "localCol: %u, "
            //            "colBlockIdCurrentRowPanel: %u, "
            //            "numColBlocksCurrentRowPanel: %u, "
            //            "startIndexOfBlockValuesCurrentBlock: %u, "
            //            "rowPanelId: %u, "
            //            "blockIdx.x: %u, "
            //            "warpId: %u, "
            //            "laneId: %u, "
            //            "idxOfFragment = %d, "
            //            "\n",
            //            idxOfMatrixP, idxOfFragment, accFrag.x[idxOfFragment],
            //            localRow, localCol, colBlockIdCurrentRowPanel,
            //            numColBlocksCurrentRowPanel,
            //            startIndexOfBlockValuesCurrentBlock,
            //            rowPanelId,
            //            blockIdx.x,
            //            warpId, laneId,
            //            idxOfFragment);
            // }
        }
    }
}

// m16n16k8
// 一个线程块负责一个行面板
// 解决 Load imbalance的问题
__global__ void sddmm_gpu_dense_block_rowPanel_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = WMMA_K;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = sddmm_dense_block_number_of_warps_per_thread_block;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startBlockId = __ldg(&blockOffsets[rowPanelId]);
    const UIN endBlockId = __ldg(&blockOffsets[rowPanelId + 1]);

    if (endBlockId - startBlockId <= 0){
        return;
    }
    const UIN endIndexOfDenseColsCurrentRowPanel = endBlockId * BLOCK_COL_SIZE;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE> accFrag;


    // Load matrix A into shared memory
#pragma unroll
    for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? __ldg(&reorderedRows[reorderedRowIndex])
                               : M;
        const UIN aColId = laneId;

        aTileSMEM[smemRow * aTileSMEMLd + laneId] =
            (aRowId < M && aColId < K)
                ? __ldg(&matrixA[aRowId * K + aColId])
                : static_cast<MATRIX_A_TYPE>(0.0f);
    }

    __syncthreads();

    for (int colBlockId = startBlockId + warpId;
         colBlockId < endBlockId;
         colBlockId += numWarps){
        fill_fragment(accFrag, 0.0f);

        const UIN startIndexOfBlockValuesCurrentBlock = colBlockId * BLOCK_SIZE;
        const UIN indexOfDenseColsCurrentColBlock = BLOCK_COL_SIZE * colBlockId;

        // Load matrix B into shared memory
#pragma unroll
        for (int iter = 0; iter < 4; ++iter){
            const UIN localK = iter * WMMA_K;
            const UIN bRowId = localK + (laneId % 2) * 4;
            const UIN reorderedColIndex =
                indexOfDenseColsCurrentColBlock + laneId / 2;
            const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentRowPanel
                                   ? __ldg(&denseCols[reorderedColIndex])
                                   : N;
            const float4 bData = (bRowId < K && bColId < N)
                                     ? __ldg(reinterpret_cast<const float4*>(
                                         &matrixB[bColId * K + bRowId]))
                                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const UIN smemOffset =
                (warpId * WMMA_N + laneId / 2) * bTileSMEMLd + (laneId % 2) * 4;
            *((float4*)(&bTileSMEM[smemOffset])) = bData;

            wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd,
                                   bTileSMEMLd);

            // Convert to TF32
#pragma unroll
            for (int i = 0; i < aFrag.num_elements; ++i)
                aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
            for (int i = 0; i < bFrag.num_elements; ++i)
                bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }

        // Store the result
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                __ldg(&blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol]);

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// 一个warp负责row panel中的1个col block
// bTileSMEMLd = WMMA_K
__global__ void sddmm_gpu_dense_block_m16n16k8_lianxu_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ denseColOffset,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4; // 4 for padding
    constexpr int bTileSMEMLd = WMMA_K; // 4 for padding

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId];

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (blockOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock =
        denseColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = denseColOffset[rowPanelId + 1];

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, each thread loads 2 elements,
        // conflict-free access
#pragma unroll
        for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[smemRow * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K)
                    ? (matrixA[aRowId * K + aColId])
                    : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
            for (int iter = 0; iter < 4; ++iter){
                const UIN localK = iter * WMMA_K;
                const UIN bRowId = kIter + localK + (laneId % 2) * 4;
                const UIN reorderedColIndex =
                    startIndexOfDenseColsCurrentColBlock + laneId / 2;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                       ? denseCols[reorderedColIndex]
                                       : N;
                const float4 bData = (bRowId < K && bColId < N)
                                         ? *reinterpret_cast<const float4*>(
                                             &matrixB[bColId * K + bRowId])
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const UIN smemOffset =
                    (warpId * WMMA_N + laneId / 2) * bTileSMEMLd + (laneId % 2) * 4;
                *((float4*)(&bTileSMEM[smemOffset])) = bData;

                wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd,
                                       bTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i)
                    aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i)
                    bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

                wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
// batch version
__global__ void sddmm_gpu_dense_block_batch_m16n16k8_block256(
    const UIN M,
    const UIN N,
    const UIN K,
    const UIN nnz,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;
    constexpr int number_of_tiles_loaded_in_one_cycle = kStep / WMMA_K;

    const int aTileSMEMLd = (WMMA_K * number_of_tiles_loaded_in_one_cycle);
    const int bTileSMEMLd = (WMMA_K * number_of_tiles_loaded_in_one_cycle);

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int bTileSMEMSize =
        (WMMA_N * each_thread_block_counts_the_number_Of_dense_blocks) *
        bTileSMEMLd;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN batchId = blockIdx.z;
    const UIN startMatrixAData = batchId * M * K;
    const UIN startMatrixBData = batchId * N * K;
    const UIN startMatrixPData = batchId * nnz;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId];

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    const UIN colBlockIdCurrentRowPanel = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (blockOffsets[rowPanelId] + colBlockIdCurrentRowPanel) * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock =
        blockOffsets[rowPanelId] * BLOCK_COL_SIZE + BLOCK_COL_SIZE * colBlockIdCurrentRowPanel;
    const UIN endIndexOfDenseColsCurrentPanel = blockOffsets[rowPanelId + 1] * BLOCK_COL_SIZE;

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, each thread loads 2 elements,
        // conflict-free access
#pragma unroll
        for (int iter = 0; iter < 2; ++iter){
            const UIN reorderedRowIndex =
                (rowPanelId * ROW_PANEL_SIZE) + (warpId * 2) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[(warpId * 2 + iter) * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K)
                    ? (matrixA[startMatrixAData + aRowId * K + aColId])
                    : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockIdCurrentRowPanel < numColBlocksCurrentRowPanel){
            // Load matrix B into shared memory, each thread loads 16 elements,
            // conflict-free access
#pragma unroll 8
            for (int iter = 0; iter < 16; ++iter){
                const UIN bRowId = kIter + laneId;
                const UIN reorderedColIndex =
                    startIndexOfDenseColsCurrentColBlock + iter;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                       ? denseCols[reorderedColIndex]
                                       : N;

                bTileSMEM[(warpId * WMMA_N + iter) * bTileSMEMLd + laneId] =
                    (bRowId < K && bColId < N)
                        ? matrixB[startMatrixBData + bRowId + bColId * K]
                        : static_cast<MATRIX_B_TYPE>(0.0f);
            }

            // Compute the matrix multiplication
#pragma unroll
            for (int iter = 0; iter < 32; iter += WMMA_K){
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter, aTileSMEMLd);
                wmma::load_matrix_sync(bFrag,
                                       bTileSMEM + warpId * WMMA_N * bTileSMEMLd + iter,
                                       bTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i)
                    aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i)
                    bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockIdCurrentRowPanel < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[startMatrixPData + idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

__device__ void m16n16k8_block128_double_buffer_load_matrixA(
    const UIN matrixLd,
    const UIN kIter,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const UIN endIndex,
    const UIN rowPanelId,
    const UIN* __restrict__ reorderedRows,
    const int writeStage,
    const int smemLd,
    MATRIX_A_TYPE* aTileSMEM){
    if (kIter >= matrixLd){
        return;
    }

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN localRow_smem =
        warpId * 4 + (laneId >> 3); // shared memory location. laneId / 8
    const UIN localCol_smem =
        writeStage * WMMA_K + (laneId & 7); // shared memory location. laneId % 8

    const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + localRow_smem;
    const UIN localCol_gmem = kIter + (laneId & 7); // laneId % 8

    aTileSMEM[localRow_smem * smemLd + localCol_smem] =
        (reorderedRowIndex < endIndex && localCol_gmem < matrixLd)
            ? matrixA[reorderedRows[reorderedRowIndex] * matrixLd + localCol_gmem]
            : static_cast<MATRIX_A_TYPE>(0);
}

__device__ void m16n16k8_block128_double_buffer_load_matrixB(
    const UIN matrixLd,
    const UIN kIter,
    const MATRIX_B_TYPE* __restrict__ __align__(16) matrixB,
    const UIN startIndex,
    const UIN endIndex,
    const UIN* __restrict__ reorderedCols,
    const int smemLd,
    MATRIX_B_TYPE* bTileSMEM){
    if (kIter >= matrixLd){
        return;
    }

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN localRow =
        warpId * WMMA_N + (laneId >> 1); // shared memory location. laneId / 2
    const UIN localCol =
        (laneId & 1) * 4; // shared memory location. (laneId % 2) * 4

    const UIN reorderedColIndex = startIndex + localRow;
    const float4 bData =
        (reorderedColIndex < endIndex && kIter + localCol < matrixLd)
            ? *(float4*)&(matrixB)[reorderedCols[reorderedColIndex] * matrixLd +
                kIter + localCol]
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    *(float4*)&bTileSMEM[localRow * smemLd + localCol] = bData;
}

// m16n16k8
// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [128,1,1]
// https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html#
__global__ void sddmm_gpu_dense_block_m16n16k8_block128_double_buffer(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ __align__(16) matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ reorderedCols,
    const UIN* __restrict__ reorderedColOffset,
    const UIN* __restrict__ blockRowOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr UIN aTileSMEMLd = (WMMA_K * 2 + 4); // Double buffer and 4 padding
    constexpr UIN bTileSMEMLd = (WMMA_K + 4); // 4 padding

    constexpr UIN aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr UIN bTileSMEMSize = (WMMA_N * 4) * bTileSMEMLd;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter = blockIdx.y * 4;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ __align__(16) MATRIX_B_TYPE bTileSMEM[bTileSMEMSize]; // col major

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentThreadBlock =
        reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockIter;
    const UIN endIndexOfReorderedColsCurrentPanel =
        reorderedColOffset[rowPanelId + 1];

    // Load first buffer
    m16n16k8_block128_double_buffer_load_matrixA(K, 0, matrixA, numNonZeroRow,
                                                 rowPanelId, reorderedRows, 0,
                                                 aTileSMEMLd, aTileSMEM);

    int writeStage = 1;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K){
        // Load next buffer
        m16n16k8_block128_double_buffer_load_matrixA(
            K, kIter + WMMA_K, matrixA, numNonZeroRow, rowPanelId, reorderedRows,
            writeStage, aTileSMEMLd, aTileSMEM);

        // Load matrix B tile into shared memory and compute the matrix
        // multiplication
        if (colBlockId < numColBlocksCurrentRowPanel){
            // Load matrix B tile into shared memory
            m16n16k8_block128_double_buffer_load_matrixB(
                K, kIter, matrixB, startIndexOfReorderedColsCurrentThreadBlock,
                endIndexOfReorderedColsCurrentPanel, reorderedCols, bTileSMEMLd,
                bTileSMEM);

            // load matrix A and B tile into fragment
            wmma::load_matrix_sync(aFrag, aTileSMEM + (writeStage ^ 1) * WMMA_K,
                                   aTileSMEMLd);
            wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * WMMA_N * bTileSMEMLd,
                                   bTileSMEMLd);

            // Convert to TF32
#pragma unroll
            for (int i = 0; i < aFrag.num_elements; ++i)
                aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
            for (int i = 0; i < bFrag.num_elements; ++i)
                bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }

        __syncthreads();

        writeStage ^= 1;
    }

    // Store the result
    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

__device__ int getSourceLane(int row, int col){
    return row * 2 + ((col / 4) % 2);
}

__device__ float getValue(int col, float x, float y, float z, float w){
    switch (col % 4){
    case 0:
        return x;
    case 1:
        return y;
    case 2:
        return z;
    case 3:
        return w;
    }
}

__device__ float4 shuffleBFragment(const float4& bData, int laneId){
    int groupId = laneId / 8;
    int localId = laneId % 8;

    int row0 = groupId;
    int row1 = groupId + 4;

    int col0 = localId;
    int col1 = localId + 4;

    float4 result;
    result.x = __shfl_sync(0xffffffff,
                           getValue(col0, bData.x, bData.y, bData.z, bData.w),
                           getSourceLane(row0, col0));
    result.y = __shfl_sync(0xffffffff,
                           getValue(col1, bData.x, bData.y, bData.z, bData.w),
                           getSourceLane(row0, col1));
    result.z = __shfl_sync(0xffffffff,
                           getValue(col0, bData.x, bData.y, bData.z, bData.w),
                           getSourceLane(row1, col0));
    result.w = __shfl_sync(0xffffffff,
                           getValue(col1, bData.x, bData.y, bData.z, bData.w),
                           getSourceLane(row1, col1));
    return result;
}

// m16n16k8
// blockDim: [256, 1, 1]
// 一个thread block负责一个row panel中的8个col block
__global__ void
sddmm_gpu_dense_block_m16n16k8_block256_noSMEM_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ denseCols,
    const UIN* __restrict__ denseColOffset,
    const UIN* __restrict__ blockOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEMLd = kStep + 4;
    constexpr int bTileSMEMLd = kStep;

    constexpr int aTileSMEMSize = WMMA_M * aTileSMEMLd;
    constexpr int numWarps = each_thread_block_counts_the_number_Of_dense_blocks;

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        blockOffsets[rowPanelId + 1] - blockOffsets[rowPanelId];

    const UIN colBlockIter =
        blockIdx.y * each_thread_block_counts_the_number_Of_dense_blocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        accFrag;

    fill_fragment(accFrag, 0.0f);

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (blockOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfDenseColsCurrentColBlock =
        denseColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfDenseColsCurrentPanel = denseColOffset[rowPanelId + 1];

    // Loop over K, one iteration 32
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, each thread loads 2 elements,
        // conflict-free access
#pragma unroll
        for (UIN smemRow = warpId; smemRow < WMMA_M; smemRow += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRow;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;

            aTileSMEM[smemRow * aTileSMEMLd + laneId] =
                (aRowId < M && aColId < K)
                    ? __ldg(&matrixA[aRowId * K + aColId])
                    : static_cast<MATRIX_A_TYPE>(0.0f);
        }

        __syncthreads();

        if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
            for (int iter = 0; iter < 4; ++iter){
                const UIN localK = iter * WMMA_K;

                // Load matrix B
                const UIN bRowId = kIter + localK + (laneId % 2) * 4;
                const UIN reorderedColIndex =
                    startIndexOfDenseColsCurrentColBlock + laneId / 2;
                const UIN bColId = reorderedColIndex < endIndexOfDenseColsCurrentPanel
                                       ? denseCols[reorderedColIndex]
                                       : N;
                const float4 bData = (bRowId < K && bColId < N)
                                         ? __ldg(reinterpret_cast<const float4*>(
                                             &matrixB[bColId * K + bRowId]))
                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 fragment = shuffleBFragment(bData, laneId);
                bFrag.x[0] = wmma::__float_to_tf32(fragment.x);
                bFrag.x[1] = wmma::__float_to_tf32(fragment.y);
                bFrag.x[2] = wmma::__float_to_tf32(fragment.z);
                bFrag.x[3] = wmma::__float_to_tf32(fragment.w);

                wmma::load_matrix_sync(aFrag, aTileSMEM + localK, aTileSMEMLd);

                // Convert to TF32
#pragma unroll
                for (int i = 0; i < aFrag.num_elements; ++i)
                    aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
#pragma unroll
                for (int i = 0; i < bFrag.num_elements; ++i)
                    bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);

                wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < accFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = accFrag.x[idxOfFragment];
            }
        }
    }
}

// m16n16k8
// blockDim: [512, 1, 1]
// 一个thread block负责一个row panel中的16个col block
__global__ void
sddmm_gpu_dense_block_m16n16k16_block512_matrixA_rowMaj_matrixB_colMaj(
    const UIN M,
    const UIN N,
    const UIN K,
    const MATRIX_A_TYPE* __restrict__ matrixA,
    const MATRIX_B_TYPE* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ reorderedCols,
    const UIN* __restrict__ reorderedColOffset,
    const UIN* __restrict__ blockRowOffsets,
    const UIN* __restrict__ blockValues,
    MATRIX_C_TYPE* matrixP){
    constexpr int eachThreadBlockCountsTheNumberOfColBlocks = 16;

    constexpr int aTileSMEMSize = (WMMA_M * WMMA_N) * 2;
    constexpr int bTileSMEMSize =
        (WMMA_K * WMMA_N * eachThreadBlockCountsTheNumberOfColBlocks) * 2;

    __shared__ MATRIX_A_TYPE aTileSMEM[aTileSMEMSize];
    __shared__ MATRIX_B_TYPE bTileSMEM[bTileSMEMSize];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIX_A_TYPE_FRAGMENT,
                   wmma::row_major>
        aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIX_B_TYPE_FRAGMENT,
                   wmma::col_major>
        bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIX_C_TYPE>
        cFrag;

    fill_fragment(cFrag, 0.0f);

    const UIN laneId = threadIdx.x & 31;
    const UIN warpId = threadIdx.x >> 5;

    const UIN rowPanelId = blockIdx.x;
    const UIN numColBlocksCurrentRowPanel =
        blockRowOffsets[rowPanelId + 1] - blockRowOffsets[rowPanelId];

    const UIN colBlockIter =
        blockIdx.y * eachThreadBlockCountsTheNumberOfColBlocks;
    if (colBlockIter >= numColBlocksCurrentRowPanel){
        return;
    }

    const UIN colBlockId = colBlockIter + warpId;
    const UIN startIndexOfBlockValuesCurrentBlock =
        (blockRowOffsets[rowPanelId] + colBlockId) * BLOCK_SIZE;

    const UIN startIndexOfReorderedColsCurrentColBlock =
        reorderedColOffset[rowPanelId] + BLOCK_COL_SIZE * colBlockId;
    const UIN endIndexOfReorderedColsCurrentPanel =
        reorderedColOffset[rowPanelId + 1];

    const UIN lda = K;
    const UIN ldb = K;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += WMMA_K * 2){
        // Load matrix A into shared memory, each thread loads 1 element,
        // conflict-free access
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + warpId;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? reorderedRows[reorderedRowIndex]
                               : M;
        const UIN aColId = kIter + laneId;

        aTileSMEM[warpId * 32 + laneId] = (aRowId < M && aColId < K)
                                              ? matrixA[aRowId * lda + aColId]
                                              : static_cast<MATRIX_A_TYPE>(0);

        // Load matrix B data into shared memory, each thread loads 16 elements,
        // conflict-free access
#pragma unroll
        for (int iter = 0; iter < 16; ++iter){
            const UIN bRowId = kIter + laneId;
            const UIN reorderedColIndex =
                startIndexOfReorderedColsCurrentColBlock + iter;
            const UIN bColId = reorderedColIndex < endIndexOfReorderedColsCurrentPanel
                                   ? reorderedCols[reorderedColIndex]
                                   : N;

            bTileSMEM[warpId * 512 + iter * 32 + laneId] =
                (bRowId < K && bColId < N)
                    ? matrixB[bRowId + bColId * ldb]
                    : static_cast<MATRIX_B_TYPE>(0);
        }
        __syncthreads();

        // Compute the matrix multiplication
        for (int iter = 0; iter < 2; ++iter){
            if (colBlockId < numColBlocksCurrentRowPanel){
                wmma::load_matrix_sync(aFrag, aTileSMEM + iter * WMMA_K, WMMA_K * 2);
                wmma::load_matrix_sync(bFrag, bTileSMEM + warpId * 512 + iter * WMMA_K,
                                       WMMA_K * 2);
                wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            }
        }

        __syncthreads();
    }

    // Store the result
    if (colBlockId < numColBlocksCurrentRowPanel){
#pragma unroll
        for (int idxOfFragment = 0; idxOfFragment < cFrag.num_elements;
             ++idxOfFragment){
            UIN localRow, localCol;
            calculateMatrixCFragmentCoordinates(laneId, idxOfFragment, localRow,
                                                localCol);

            const UIN idxOfMatrixP =
                blockValues[startIndexOfBlockValuesCurrentBlock +
                    localRow * BLOCK_COL_SIZE + localCol];

            // Saved when the value is not 0
            if (idxOfMatrixP != NULL_VALUE){
                matrixP[idxOfMatrixP] = cFrag.x[idxOfFragment];
            }
        }
    }
}

// matrixA_rowMajor
// matrixB_colMajor
__global__ void sddmm_gpu_sparse_block_2threadOneData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseValueOffsets,
    const UIN* __restrict__ sparseValues,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseCols,
    float* matrixP){
    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize =
        WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock =
        sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseValueOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel = sparseValueOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel){
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + (tId >> 1);

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseCols[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段,
    // 避免了线程之间的数据竞争
    const UIN oddOrEven = laneId & 1;

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (UIN iter = warpId; iter < WMMA_M; iter += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[iter * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K)
                    ? __ldg(&matrixA[aRowId * K + aColId])
                    : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate
        // one element
        if (index < indexBoundaryCurrentRowPanel){
#pragma unroll
            for (int localKIter = oddOrEven * kStepPerThread;
                 localKIter < (oddOrEven + 1) * kStepPerThread; localKIter += 4){
                const float4 aData =
                    *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 bData =
                    __ldg((float4*)&matrixB[col * K + kIter + localKIter]);
                c += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z +
                    aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    // Use the shuffle instruction to merge the results of two adjacent threads.
    const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
    c += __shfl_xor_sync(mask, c,
                         1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上,
    // 线程1的sm1加到线程0的sm1上

    if (index < indexBoundaryCurrentRowPanel && oddOrEven == 0){
        matrixP[sparseValues[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// 解决 Load imbalance 的问题
__global__ void sddmm_gpu_sparse_block_2_2threadOneData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseValueOffsets,
    const UIN* __restrict__ sparseValues,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseCols,
    const UIN* __restrict__ rowPanelIds,
    const UIN* __restrict__ colBlockIters,
    float* matrixP){
    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize =
        WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock =
        sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    // const UIN startIndexOfSparseDataCurrentBlock =
    //     blockIdx.x * calculateDataPerThreadBlock;
    const UIN rowPanelId = rowPanelIds[blockIdx.x];
    const UIN startIndexOfSparseDataCurrentBlock = sparseValueOffsets[rowPanelId] + colBlockIters[blockIdx.x];

    // while (startIndexOfSparseDataCurrentBlock >=
    //     sparseValueOffsets[rowPanelId + 1]){
    //     ++rowPanelId;
    // }

    const UIN indexBoundaryCurrentRowPanel = sparseValueOffsets[rowPanelId + 1];

    const UIN index = startIndexOfSparseDataCurrentBlock + (tId >> 1);

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseCols[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段,
    // 避免了线程之间的数据竞争
    const UIN oddOrEven = laneId & 1;

    float c0 = 0;
    // float c1 = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (UIN iter = warpId; iter < WMMA_M; iter += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[iter * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K)
                    ? __ldg(&matrixA[aRowId * K + aColId])
                    : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate
        // one element
        if (index < indexBoundaryCurrentRowPanel){
#pragma unroll
            for (int localKIter = oddOrEven * kStepPerThread;
                 localKIter < (oddOrEven + 1) * kStepPerThread; localKIter += 8){
                // 增加指令级并行(ILP)
                const float4 aData0 =
                    *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 aData1 =
                    *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 4]);

                const float4 bData0 =
                    __ldg((float4*)&matrixB[col * K + kIter + localKIter]);
                const float4 bData1 =
                    __ldg((float4*)&matrixB[col * K + kIter + localKIter + 4]);

                c0 += aData0.x * bData0.x + aData0.y * bData0.y + aData0.z * bData0.z + aData0.w * bData0.w +
                    aData1.x * bData1.x + aData1.y * bData1.y + aData1.z * bData1.z + aData1.w * bData1.w;
                // c1 += aData1.x * bData1.x + aData1.y * bData1.y + aData1.z * bData1.z + aData1.w * bData1.w;
            }
        }

        __syncthreads();
    }

    // Use the shuffle instruction to merge the results of two adjacent threads.
    const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
    c0 += __shfl_xor_sync(mask, c0, 1); // 使用shuffle指令. 使相邻的线程的c0结果相加
    // c1 += __shfl_xor_sync(mask, c1, 1); // 使用shuffle指令. 使相邻的线程的c1结果相加


    if (index < indexBoundaryCurrentRowPanel && oddOrEven == 0){
        matrixP[sparseValues[index]] = c0;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// 一个线程块计算一个行面板
__global__ void sddmm_gpu_sparse_remainder_k32_2threadOneData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseValueOffsets,
    const UIN* __restrict__ sparseValues,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseCols,
    float* matrixP){
    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize = WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseData = sparseValueOffsets[rowPanelId];
    const UIN endIndexOfSparseData = sparseValueOffsets[rowPanelId + 1];

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段,
    const UIN oddOrEven = laneId & 1;

    // Load matrix A into shared memory, conflict-free access
#pragma unroll
    for (UIN iter = warpId; iter < WMMA_M; iter += numWarps){
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? reorderedRows[reorderedRowIndex]
                               : M;
        const UIN aColId = laneId;
        aTileSMEM[iter * aTileSMEM_ld + laneId] =
            (aRowId < M && aColId < K)
                ? __ldg(&matrixA[aRowId * K + aColId])
                : static_cast<float>(0);
    }
    __syncthreads();

    for (int idx = startIndexOfSparseData + (tId >> 1); idx < endIndexOfSparseData; idx += (blockDim.x >> 1)){
        float c0 = 0;
        float c1 = 0;
        const UIN relativeRow = relativeRows[idx];
        const UIN col = sparseCols[idx];
        // Load matrix B and compute the matrix multiplication, 2 thread calculate
#pragma unroll
        for (int localKIter = oddOrEven * kStepPerThread;
             localKIter < (oddOrEven + 1) * kStepPerThread; localKIter += 8){
            // 增加指令级并行(ILP)
            const float4 bData0 = __ldg((float4*)&matrixB[col * K + localKIter]);
            const float4 bData1 = __ldg((float4*)&matrixB[col * K + localKIter + 4]);

            c0 += aTileSMEM[relativeRow * aTileSMEM_ld + localKIter] * bData0.x +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 1] * bData0.y +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 2] * bData0.z +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 3] * bData0.w;

            c1 += aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 4] * bData1.x +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 5] * bData1.y +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 6] * bData1.z +
                aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 7] * bData1.w;

            // const float4 aData0 = *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
            // const float4 aData1 = *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter + 4]);
            //
            // c0 += aData0.x * bData0.x + aData0.y * bData0.y + aData0.z * bData0.z + aData0.w * bData0.w;
            // c1 += aData1.x * bData1.x + aData1.y * bData1.y + aData1.z * bData1.z + aData1.w * bData1.w;
        }

        // Use the shuffle instruction to merge the results of two adjacent threads.
        const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
        c0 += __shfl_xor_sync(mask, c0, 1); // 使用shuffle指令. 使相邻的线程的c0结果相加
        c1 += __shfl_xor_sync(mask, c1, 1); // 使用shuffle指令. 使相邻的线程的c1结果相加

        if (oddOrEven == 0){
            matrixP[sparseValues[idx]] = c0 + c1;
        }
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// batch version
__global__ void sddmm_gpu_sparse_block_batch_2threadOneData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const UIN nnz,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseValueOffsets,
    const UIN* __restrict__ sparseValues,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseCols,
    float* matrixP){
    constexpr int kStep = 32;
    constexpr int kStepPerThread = kStep / 2;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize =
        WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    constexpr int calculateDataPerThreadBlock =
        sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;

    const UIN batchId = blockIdx.z;
    const UIN startMatrixAData = batchId * M * K;
    const UIN startMatrixBData = batchId * N * K;
    const UIN startMatrixPData = batchId * nnz;

    const UIN numWarps = blockDim.x / WARP_SIZE;

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseValueOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel = sparseValueOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel){
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + (tId >> 1);

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseCols[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    // 如果tid是偶数则是0; 如果tid是奇数则是1. 确保不同线程并行处理不同的数据段,
    // 避免了线程之间的数据竞争
    const UIN oddOrEven = laneId & 1;

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (UIN iter = warpId; iter < WMMA_M; iter += numWarps){
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + iter;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[iter * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K)
                    ? (matrixA[startMatrixAData + aRowId * K + aColId])
                    : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate
        // one element
        if (index < indexBoundaryCurrentRowPanel){
#pragma unroll
            for (int localKIter = oddOrEven * kStepPerThread;
                 localKIter < (oddOrEven + 1) * kStepPerThread; localKIter += 4){
                const float4 aData = *((float4*)&aTileSMEM[relativeRow * aTileSMEM_ld + localKIter]);
                const float4 bData = *((float4*)&matrixB[startMatrixBData + col * K + kIter + localKIter]);
                c += aData.x * bData.x + aData.y * bData.y + aData.z * bData.z +
                    aData.w * bData.w;
            }
        }

        __syncthreads();
    }

    // Use the shuffle instruction to merge the results of two adjacent threads.
    const unsigned mask = (1 << tId) | (1 << (tId ^ 1)); // 只同步相邻线程
    c += __shfl_xor_sync(mask, c,
                         1); // 使用shuffle指令. 使线程0的sm1加到线程1的sm1上,
    // 线程1的sm1加到线程0的sm1上

    if (index < indexBoundaryCurrentRowPanel && oddOrEven == 0){
        matrixP[startMatrixPData + sparseValues[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_block512_warpOneData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseDataOffsets,
    const UIN* __restrict__ sparseData,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseColIndices,
    float* matrixP){
    // 线程块中线程数量
    constexpr int numWarpsPerBlock = 16;
    constexpr UIN calculateDataPerThreadBlock = numWarpsPerBlock;

    constexpr int kStep = 32;

    constexpr int aTileSMEM_ld = kStep + 4; // 4 padding
    constexpr int aTileSMEMSize =
        WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] + blockIdx.y * calculateDataPerThreadBlock;
    const UIN indexBoundaryCurrentRowPanel = sparseDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel){
        return;
    }

    const UIN index = startIndexOfSparseDataCurrentBlock + warpId;

    const UIN relativeRow = relativeRows[index];
    const UIN col = sparseColIndices[index];

    __shared__ float aTileSMEM[aTileSMEMSize];

    float c = 0;

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, conflict-free access
        const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + warpId;
        const UIN aRowId = reorderedRowIndex < numNonZeroRow
                               ? reorderedRows[reorderedRowIndex]
                               : M;
        const UIN aColId = kIter + laneId;
        aTileSMEM[warpId * aTileSMEM_ld + laneId] =
            (aRowId < M && aColId < K)
                ? matrixA[aRowId * K + aColId]
                : static_cast<float>(0);
        __syncthreads();

        // Load matrix B and compute the matrix multiplication, 2 thread calculate
        // one element
        if (index < indexBoundaryCurrentRowPanel){
            const float aData = aTileSMEM[relativeRow * aTileSMEM_ld + laneId];
            const float bData = matrixB[col * K + kIter + laneId];
            c += aData * bData;
        }

        __syncthreads();
    }

    c = cuUtil::warp_reduce_sum(c);

    if (index < indexBoundaryCurrentRowPanel && laneId == 0){
        matrixP[sparseData[index]] = c;
    }
}

// matrixA_rowMajor
// matrixB_colMajor
// blockDim: [512,1,1]
__global__ void sddmm_gpu_sparse_block_warpMutData_shuffle(
    const UIN M,
    const UIN N,
    const UIN K,
    const float* __restrict__ matrixA,
    const float* __restrict__ matrixB,
    const UIN numNonZeroRow,
    const UIN* __restrict__ reorderedRows,
    const UIN* __restrict__ sparseDataOffsets,
    const UIN* __restrict__ sparseData,
    const UIN* __restrict__ relativeRows,
    const UIN* __restrict__ sparseColIndices,
    float* matrixP){
    constexpr int kStep = 32;

    constexpr int aTileSMEM_ld = kStep;
    constexpr int aTileSMEMSize =
        WMMA_M * aTileSMEM_ld; // 512 actual data and 64 padding

    const UIN tId = threadIdx.x;

    const UIN laneId = tId & 31;
    const UIN warpId = tId >> 5;

    const UIN numWarps = (blockDim.x + 31) >> 5;

    const UIN calculateDataPerWarp =
        sddmm_sparse_block_each_thread_block_counts_the_number_Of_data / numWarps;

    const UIN rowPanelId = blockIdx.x;

    const UIN startIndexOfSparseDataCurrentBlock =
        sparseDataOffsets[rowPanelId] +
        blockIdx.y *
        sddmm_sparse_block_each_thread_block_counts_the_number_Of_data;
    const UIN indexBoundaryCurrentRowPanel = sparseDataOffsets[rowPanelId + 1];

    // If the current block is out of the boundary, return
    if (startIndexOfSparseDataCurrentBlock >= indexBoundaryCurrentRowPanel){
        return;
    }

    __shared__ float aTileSMEM[aTileSMEMSize];
    extern __shared__ float
        pSMEM[]; // sddmm_sparse_block_each_thread_block_counts_the_number_Of_data

    // Loop over K
    for (int kIter = 0; kIter < K; kIter += kStep){
        // Load matrix A into shared memory, conflict-free access
#pragma unroll
        for (int rowIter = 0; rowIter < WMMA_M; rowIter += numWarps){
            const UIN smemRowId = rowIter + warpId;
            const UIN reorderedRowIndex = (rowPanelId * ROW_PANEL_SIZE) + smemRowId;
            const UIN aRowId = reorderedRowIndex < numNonZeroRow
                                   ? reorderedRows[reorderedRowIndex]
                                   : M;
            const UIN aColId = kIter + laneId;
            aTileSMEM[smemRowId * aTileSMEM_ld + laneId] =
                (aRowId < M && aColId < K)
                    ? matrixA[aRowId * K + aColId]
                    : static_cast<float>(0);
        }
        __syncthreads();

        // Load matrix B and compute the matrix multiplication
#pragma unroll
        for (int iter = 0; iter < calculateDataPerWarp; ++iter){
            const UIN index = startIndexOfSparseDataCurrentBlock +
                warpId * calculateDataPerWarp + iter;
            const UIN relativeRow = relativeRows[index];
            const UIN col = sparseColIndices[index];
            const float aData = aTileSMEM[relativeRow * aTileSMEM_ld + laneId];
            const float bData = matrixB[col * K + kIter + laneId];
            float c = aData * bData;

            c = cuUtil::warp_reduce_sum(c);
            if (laneId == 0){
                pSMEM[warpId * calculateDataPerWarp + iter] += c;
            }
        }

        __syncthreads();
    }

    if (tId < sddmm_sparse_block_each_thread_block_counts_the_number_Of_data){
        const UIN index = startIndexOfSparseDataCurrentBlock + tId;
        matrixP[sparseData[index]] = pSMEM[tId];
    }
}

#define TILE_DIM 32

__global__ void batchedMatrixTranspose(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int width,
                                       int height,
                                       int batchStrideIn,
                                       int batchStrideOut){
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // 避免 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int batchIdx = blockIdx.z;

    const float* batchInput = input + batchIdx * batchStrideIn;
    float* batchOutput = output + batchIdx * batchStrideOut;

    // Load input tile into shared memory
    if (x < width && y < height){
        tile[threadIdx.y][threadIdx.x] = batchInput[y * width + x];
    }

    __syncthreads();

    // Transpose indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width){
        batchOutput[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
} // namespace kernel

void sddmm_gpu(const Matrix<float>& matrixA,
               const Matrix<float>& matrixB,
               const RPHM& rphm,
               sparseMatrix::CSR<float>& matrixP,
               Logger& logger){
    dev::vector<float> matrixA_dev(matrixA.values());
    dev::vector<float> matrixB_dev(matrixB.values());
    dev::vector<float> matrixP_dev(matrixP.nnz(), 0);

    if (matrixA.col() <= 32){
        sddmm_gpu_k32(matrixP.row(), matrixP.col(), matrixA.col(), matrixA_dev.data(),
                      matrixB_dev.data(), rphm, matrixP_dev.data(), logger);
    }
    else{
        sddmm_gpu(matrixP.row(), matrixP.col(), matrixA.col(), matrixA_dev.data(),
                  matrixB_dev.data(), rphm, matrixP_dev.data(), logger);
    }

    // Copy the results from the device to the host
    matrixP.setValues() = d2h(matrixP_dev);
}

void sddmm_gpu(UIN M,
               UIN N,
               UIN K,
               const float* matrixA,
               const float* matrixB,
               const RPHM& rphm,
               float* matrixP,
               Logger& logger){
    //    // Convert the data type of matrix A and matrix B for use tensor core
    //    dev::vector<MATRIX_A_TYPE> matrixA_convertedType(M * K);
    //    dev::vector<MATRIX_B_TYPE> matrixB_convertedType(N * K);
    //    {
    //        const int numThreadPerBlock = 1024;
    //        kernel::convertDataType<<< (M * K + numThreadPerBlock - 1) /
    //        numThreadPerBlock, numThreadPerBlock>>>(
    //            M * K, matrixA, matrixA_convertedType.data());
    //        kernel::convertDataType<<< (N * K + numThreadPerBlock - 1) /
    //        numThreadPerBlock, numThreadPerBlock>>>(
    //            N * K, matrixB, matrixB_convertedType.data());
    //    }

    dim3 grid_dense, block_dense, grid_sparse, block_sparse;

    block_dense.x = WARP_SIZE *
        sddmm_dense_block_number_of_warps_per_thread_block;
    // Assign row panel to x-axis of grid, and assign col block to y-axis of grid
    grid_dense.x = rphm.numRowPanels();
    grid_dense.y =
        std::ceil(static_cast<float>(rphm.maxNumDenseColBlocksInRowPanel())
            /
            each_thread_block_counts_the_number_Of_dense_blocks);

    // block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
    // grid_sparse.x = rphm.numRowPanels();
    // grid_sparse.y = rphm.maxNumSparseColBlocksInRowPanel();

    // block_dense.x =
    //     WARP_SIZE * sddmm_dense_block_number_of_warps_per_thread_block;
    // grid_dense.x = rphm.numDenseThreadBlocks();

    block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
    grid_sparse.x = rphm.numSparseThreadBlocks();

    // printf("grid_dense: [%u, %u, %u], block_dense: [%u, %u, %u]\n", grid_dense.x,
    //        grid_dense.y, grid_dense.z, block_dense.x, block_dense.y,
    //        block_dense.z);
    // printf("grid_sparse: [%u, %u, %u], block_sparse: [%u, %u, %u]\n",
    //        grid_sparse.x, grid_sparse.y, grid_sparse.z, block_sparse.x,
    //        block_sparse.y, block_sparse.z);

    cudaStream_t denseStream;
    cudaStream_t sparseStream;

    cudaStreamCreate(&denseStream);
    cudaStreamCreate(&sparseStream);

    CudaTimeCalculator totalTimeCalculator;

    totalTimeCalculator.startClock();

    for (int iter = 0; iter < logger.numITER_; ++iter){
#ifdef WMMA_16_16_8
        if (grid_dense.x > 0 && grid_dense.y > 0){
            kernel::sddmm_gpu_dense_block_m16n16k8_matrixA_rowMaj_matrixB_colMaj<<<
                grid_dense, block_dense, 0, denseStream>>>(
                    M, N, K, matrixA, matrixB, rphm.reorderedRows().size(),
                    rphm.reorderedRows().data(), rphm.denseCols().data(),
                    rphm.blockOffsets().data(),
                    rphm.blockValues().data(),
                    matrixP);
            // kernel::sddmm_gpu_dense_block_2_m16n16k8_matrixA_rowMaj_matrixB_colMaj<<<
            //     grid_dense, block_dense, 0, denseStream>>>(
            //         M, N, K, matrixA, matrixB, rphm.reorderedRows().size(),
            //         rphm.reorderedRows().data(), rphm.denseCols().data(),
            //         rphm.blockOffsets().data(),
            //         rphm.blockValues().data(),
            //         rphm.denseRowPanelIds().data(),
            //         rphm.denseColBlockIters().data(),
            //         matrixP);
        }
#endif // WMMA_16_16_8

        if (grid_sparse.x > 0 && grid_sparse.y > 0){
            // kernel::sddmm_gpu_sparse_block_2threadOneData_shuffle<<<grid_sparse,
            //     block_sparse, 0, sparseStream>>>(
            //         M, N, K,
            //         matrixA,
            //         matrixB,
            //         rphm.reorderedRows().size(),
            //         rphm.reorderedRows().data(),
            //         rphm.sparseValueOffsets().data(),
            //         rphm.sparseValues().data(),
            //         rphm.sparseRelativeRows().data(),
            //         rphm.sparseColIndices().data(),
            //         matrixP);
            kernel::sddmm_gpu_sparse_block_2_2threadOneData_shuffle<<<grid_sparse,
                block_sparse, 0, sparseStream>>>(
                    M, N, K,
                    matrixA,
                    matrixB,
                    rphm.reorderedRows().size(),
                    rphm.reorderedRows().data(),
                    rphm.sparseValueOffsets().data(),
                    rphm.sparseValues().data(),
                    rphm.sparseRelativeRows().data(),
                    rphm.sparseColIndices().data(),
                    rphm.sparseRowPanelIds().data(),
                    rphm.sparseColBlockIters().data(),
                    matrixP);
            // kernel::sddmm_gpu_sparse_remainder_k32_2threadOneData_shuffle<<<grid_sparse,block_sparse, 0,
            //     sparseStream>>>(
            //         M, N, K,
            //         matrixA,
            //         matrixB,
            //         rphm.reorderedRows().size(),
            //         rphm.reorderedRows().data(),
            //         rphm.sparseValueOffsets().data(),
            //         rphm.sparseValues().data(),
            //         rphm.sparseRelativeRows().data(),
            //         rphm.sparseColIndices().data(),
            //         matrixP);
        }
    }

    totalTimeCalculator.endClock();

    const float totalTime = totalTimeCalculator.getTime();
    const float singleTime = totalTime / logger.numITER_;

    logger.gridDim_dense_ = grid_dense;
    logger.gridDim_sparse_ = grid_sparse;
    logger.blockDim_dense_ = block_dense;
    logger.blockDim_sparse_ = block_sparse;
    logger.sddmmTime_ = singleTime;

    cudaStreamDestroy(denseStream);
    cudaStreamDestroy(sparseStream);
}

void sddmm_gpu_k32(UIN M,
                   UIN N,
                   UIN K,
                   const float* matrixA,
                   const float* matrixB,
                   const RPHM& rphm,
                   float* matrixP,
                   Logger& logger){
    //    // Convert the data type of matrix A and matrix B for use tensor core
    //    dev::vector<MATRIX_A_TYPE> matrixA_convertedType(M * K);
    //    dev::vector<MATRIX_B_TYPE> matrixB_convertedType(N * K);
    //    {
    //        const int numThreadPerBlock = 1024;
    //        kernel::convertDataType<<< (M * K + numThreadPerBlock - 1) /
    //        numThreadPerBlock, numThreadPerBlock>>>(
    //            M * K, matrixA, matrixA_convertedType.data());
    //        kernel::convertDataType<<< (N * K + numThreadPerBlock - 1) /
    //        numThreadPerBlock, numThreadPerBlock>>>(
    //            N * K, matrixB, matrixB_convertedType.data());
    //    }

    dim3 grid_dense, block_dense, grid_sparse, block_sparse;

    block_dense.x = WARP_SIZE *
        sddmm_dense_block_number_of_warps_per_thread_block;
    // Assign row panel to x-axis of grid, and assign col block to y-axis of grid
    grid_dense.x = rphm.numRowPanels();
    grid_dense.y = std::ceil(
        static_cast<float>(rphm.maxNumDenseColBlocksInRowPanel()) /
        each_thread_block_counts_the_number_Of_dense_blocks);

    // block_dense.x =
    //     WARP_SIZE * sddmm_dense_block_number_of_warps_per_thread_block;
    // grid_dense.x = rphm.numDenseThreadBlocks();

    block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
    grid_sparse.x = rphm.numRowPanels();

    // printf("grid_dense: [%u, %u, %u], block_dense: [%u, %u, %u]\n", grid_dense.x,
    //        grid_dense.y, grid_dense.z, block_dense.x, block_dense.y,
    //        block_dense.z);
    // printf("grid_sparse: [%u, %u, %u], block_sparse: [%u, %u, %u]\n",
    //        grid_sparse.x, grid_sparse.y, grid_sparse.z, block_sparse.x,
    //        block_sparse.y, block_sparse.z);

    cudaStream_t denseStream;
    cudaStream_t sparseStream;

    cudaStreamCreate(&denseStream);
    cudaStreamCreate(&sparseStream);

    CudaTimeCalculator totalTimeCalculator;

    totalTimeCalculator.startClock();

    for (int iter = 0; iter < 10; ++iter){
#ifdef WMMA_16_16_8
        if (grid_dense.x > 0 && grid_dense.y > 0){
            kernel::sddmm_gpu_dense_block_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj<<<
                grid_dense, block_dense, 0, denseStream>>>(
                    M, N, K, matrixA, matrixB, rphm.reorderedRows().size(),
                    rphm.reorderedRows().data(), rphm.denseCols().data(),
                    rphm.blockOffsets().data(),
                    rphm.blockValues().data(),
                    matrixP);
            // kernel::sddmm_gpu_dense_block_2_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj<<<
            //     grid_dense, block_dense, 0, denseStream>>>(
            //         M, N, K, matrixA, matrixB, rphm.reorderedRows().size(),
            //         rphm.reorderedRows().data(), rphm.denseCols().data(),
            //         rphm.blockOffsets().data(),
            //         rphm.blockValues().data(),
            //         rphm.denseRowPanelIds().data(),
            //         rphm.denseColBlockIters().data(),
            //         matrixP);
            // kernel::sddmm_gpu_dense_block_rowPanel_k32_m16n16k8_matrixA_rowMaj_matrixB_colMaj<<<
            //     grid_dense, block_dense, 0, denseStream>>>(
            //         M, N, K, matrixA, matrixB, rphm.reorderedRows().size(),
            //         rphm.reorderedRows().data(), rphm.denseCols().data(),
            //         rphm.blockOffsets().data(),
            //         rphm.blockValues().data(),
            //         matrixP);
        }
#endif // WMMA_16_16_8

        if (grid_sparse.x > 0 && grid_sparse.y > 0){
            kernel::sddmm_gpu_sparse_remainder_k32_2threadOneData_shuffle<<<grid_sparse,
                block_sparse, 0, sparseStream>>>(
                    M, N, K,
                    matrixA,
                    matrixB,
                    rphm.reorderedRows().size(),
                    rphm.reorderedRows().data(),
                    rphm.sparseValueOffsets().data(),
                    rphm.sparseValues().data(),
                    rphm.sparseRelativeRows().data(),
                    rphm.sparseColIndices().data(),
                    matrixP);
        }
    }

    totalTimeCalculator.endClock();

    const float totalTime = totalTimeCalculator.getTime();
    const float singleTime = totalTime / logger.numITER_;

    logger.gridDim_dense_ = grid_dense;
    logger.gridDim_sparse_ = grid_sparse;
    logger.blockDim_dense_ = block_dense;
    logger.blockDim_sparse_ = block_sparse;
    logger.sddmmTime_ = singleTime;

    cudaStreamDestroy(denseStream);
    cudaStreamDestroy(sparseStream);
}

void sddmm_gpu_batch(const UIN numBatch,
                     const UIN M,
                     const UIN N,
                     const UIN K,
                     const UIN nnz,
                     const float* matrixA,
                     const float* matrixB,
                     const RPHM& rphm,
                     float* matrixP,
                     float& time){
    cudaStream_t denseStream;
    cudaStream_t sparseStream;

    cudaStreamCreate(&denseStream);
    cudaStreamCreate(&sparseStream);

    dim3 grid_dense, block_dense, grid_sparse, block_sparse;

    block_dense.x =
        WARP_SIZE * sddmm_dense_block_number_of_warps_per_thread_block;
    // Assign row panel to x-axis of grid, and assign col block to y-axis of grid
    grid_dense.x = rphm.numRowPanels();
    grid_dense.y =
        std::ceil(static_cast<float>(rphm.maxNumDenseColBlocksInRowPanel()) /
            each_thread_block_counts_the_number_Of_dense_blocks);
    grid_dense.z = numBatch;

    block_sparse.x = sddmm_sparse_block_number_of_thread_per_thread_block;
    grid_sparse.x = rphm.numRowPanels();
    grid_sparse.y = rphm.maxNumSparseColBlocksInRowPanel();
    grid_sparse.z = numBatch;

    printf("grid_dense: [%u, %u, %u], block_dense: [%u, %u, %u]\n", grid_dense.x,
           grid_dense.y, grid_dense.z, block_dense.x, block_dense.y,
           block_dense.z);
    printf("grid_sparse: [%u, %u, %u], block_sparse: [%u, %u, %u]\n",
           grid_sparse.x, grid_sparse.y, grid_sparse.z, block_sparse.x,
           block_sparse.y, block_sparse.z);

    CudaTimeCalculator totalTimeCalculator, denseKernelTimeCalculator,
                       sparseKernelTimeCalculator;

    totalTimeCalculator.startClock();

    denseKernelTimeCalculator.startClock(denseStream);

#ifdef WMMA_16_16_8
    kernel::sddmm_gpu_dense_block_batch_m16n16k8_block256<<<
        grid_dense, block_dense, 0, denseStream>>>(
            M, N, K, nnz, matrixA, matrixB, rphm.reorderedRows().size(),
            rphm.reorderedRows().data(), rphm.denseCols().data(),
            rphm.blockOffsets().data(),
            rphm.blockValues().data(), matrixP);
#endif // WMMA_16_16_8

    denseKernelTimeCalculator.endClock(denseStream);

    sparseKernelTimeCalculator.startClock(sparseStream);

    kernel::sddmm_gpu_sparse_block_batch_2threadOneData_shuffle<<<
        grid_sparse, block_sparse, 0, sparseStream>>>(
            M, N, K, nnz, matrixA, matrixB, rphm.reorderedRows().size(),
            rphm.reorderedRows().data(), rphm.sparseValueOffsets().data(),
            rphm.sparseValues().data(), rphm.sparseRelativeRows().data(),
            rphm.sparseColIndices().data(), matrixP);

    sparseKernelTimeCalculator.endClock(sparseStream);

    totalTimeCalculator.endClock();

    const float denseBlockTime = denseKernelTimeCalculator.getTime();
    const float sparseBlockTime = sparseKernelTimeCalculator.getTime();
    const float totalTime = totalTimeCalculator.getTime();

    const float overlapEfficiency =
        (denseBlockTime + sparseBlockTime) / totalTime;

    printf("denseBlockTime: %f ms\n", denseBlockTime);
    printf("sparseBlockTime: %f ms\n", sparseBlockTime);
    printf("totalTime: %f ms, overlapEfficiency: %f\n", totalTime,
           overlapEfficiency);

    time = totalTime;

    cudaStreamDestroy(denseStream);
    cudaStreamDestroy(sparseStream);
}

void batchedMatrixTranspose(const UIN width,
                            const UIN height,
                            const UIN numBatches,
                            const float* d_input,
                            float* d_output){
    // Launch
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM,
                 numBatches); // batch as grid.z

    kernel::batchedMatrixTranspose<<<gridDim, blockDim>>>(
        d_input, d_output, width, height,
        width * height, // batchStrideIn
        width * height // batchStrideOut
    );
    cudaDeviceSynchronize();
}
