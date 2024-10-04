#include <cstdio>

#include "devMatrixKernel.cuh"

template<typename T>
__global__ void getValuesFromDenseData(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                       const UIN *rowIndex, const UIN *colIndex,
                                       const T *denseData, T *output) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

}

template __global__ void getValuesFromDenseData<int>(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                                     const UIN *rowIndex, const UIN *colIndex,
                                                     const int *denseData, int *output);
template __global__ void getValuesFromDenseData<float>(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                                       const UIN *rowIndex, const UIN *colIndex,
                                                       const float *denseData, float *output);
template __global__ void getValuesFromDenseData<double>(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                                        const UIN *rowIndex, const UIN *colIndex,
                                                        const double *denseData, double *output);

__global__ void getNumIndexPerWarp_1(const UIN size, const UIN numWarpX,
                                     const UIN numTileM, const UIN numTileN,
                                     const UIN nnz,
                                     const UIN *rowIndex,
                                     const UIN *colIndex,
                                     UIN *numIndexPerWarp) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) {
        return;
    }

    const int curWarpX = tid % numWarpX;
    const int curWarpY = tid / numWarpX;
    if (curWarpX > numTileN || curWarpY > numTileM) {
        return;
    }

    const UIN rowBeginOfTile = (tid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (tid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (tid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (tid % numWarpX + 1) * WMMA_N;

    UIN num = 0;
    for (int idx = 0; idx < nnz; ++idx) {
        const UIN curRow = rowIndex[idx];
        const UIN curCol = colIndex[idx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            ++num;
        }
    }

    numIndexPerWarp[tid] = num;
}

__global__ void getTileIndexDataPerWarp(const UIN size, const UIN numWarpX,
                                        const UIN numTileM, const UIN numTileN,
                                        const UIN nnz,
                                        const UIN *rowIndex,
                                        const UIN *colIndex,
                                        const UIN *matrixTileMappedToWarpIndex,
                                        UIN *matrixTileMappedToWarpIndexData) {
    const int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalTid >= size) {
        return;
    }
    const int curWarpX = globalTid % numWarpX;
    const int curWarpY = globalTid / numWarpX;
    if (curWarpX > numTileN || curWarpY > numTileM) {
        return;
    }

    const UIN rowBeginOfTile = (globalTid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (globalTid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (globalTid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (globalTid % numWarpX + 1) * WMMA_N;

    const UIN beginIdx = matrixTileMappedToWarpIndex[globalTid];

    UIN count = 0;
    for (int idx = 0; idx < nnz; ++idx) {
        const UIN curRow = rowIndex[idx];
        const UIN curCol = colIndex[idx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            matrixTileMappedToWarpIndexData[beginIdx + count] = idx;
            ++count;
        }
    }

}

/**
 * Pipeline method
 * error
 **/
template<typename OP>
__global__ void getIndexPerWarp_2(const UIN size, const UIN numWarpX,
                                  const UIN numTileM, const UIN numTileN,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (globalTid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (globalTid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (globalTid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (globalTid % numWarpX + 1) * WMMA_N;

    __shared__ UIN rowIndexShared[SHARED_MEMORY_SIZE];
    __shared__ UIN colIndexShared[SHARED_MEMORY_SIZE];

    const int sharedBeginIdx = threadIdx.x * NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
    const int sharedEndIdx = sharedBeginIdx + NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
//    printf(" sharedBeginIdx = %d, sharedEndIdx = %d\n", sharedBeginIdx, sharedEndIdx);

    op.init(gridDim, blockIdx, blockDim, threadIdx);
    for (int loop = 0; loop < nnz; loop += SHARED_MEMORY_SIZE) {

#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
        for (int mtxIdx = loop + sharedBeginIdx, sharedIdx = sharedBeginIdx;
             mtxIdx < nnz && sharedIdx < sharedEndIdx;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
//            if (mtxIdx == nnz-1) {
//                printf("%d ", mtxIdx);
//            }
        }
        __syncthreads();
//        if (globalTid == 0 && loop + SHARED_MEMORY_SIZE > nnz) {
//            for (int idx = 0; idx < SHARED_MEMORY_SIZE; ++idx) {
//                printf("%d ",rowIndexShared[idx]);
//            }
//        }

        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
        for (int sharedIdx = 0; sharedIdx < SHARED_MEMORY_SIZE /*&& sharedIdx < sharedLoopEnd*/; ++sharedIdx) {
            const UIN curRow = rowIndexShared[sharedIdx];
            const UIN curCol = colIndexShared[sharedIdx];
            if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
                curCol >= colBeginOfTile && curCol < colEndOfTile) {
                op.cycle(loop + sharedIdx);
            }
        }
    }
    op.done();
}

template __global__ void getIndexPerWarp_2<updateNumOfIndexOperator_2>(const UIN size,
                                                                       const UIN numWarpX,
                                                                       const UIN numTileM,
                                                                       const UIN numTileN,
                                                                       const UIN nnz,
                                                                       const UIN *rowIndex,
                                                                       const UIN *colIndex,
                                                                       updateNumOfIndexOperator_2 op);
template __global__ void getIndexPerWarp_2<updateIndexDataPerWarpOperator_2>(const UIN size,
                                                                       const UIN numWarpX,
                                                                       const UIN numTileM,
                                                                       const UIN numTileN,
                                                                       const UIN nnz,
                                                                       const UIN *rowIndex,
                                                                       const UIN *colIndex,
                                                                       updateIndexDataPerWarpOperator_2 op);

__global__ void getTileIndexDataPerWarp_2(const UIN size, const UIN numWarpX,
                                          const UIN numTileM, const UIN numTileN,
                                          const UIN nnz,
                                          const UIN *rowIndex,
                                          const UIN *colIndex,
                                          const UIN *matrixTileMappedToWarpIndex,
                                          UIN *matrixTileMappedToWarpIndexData) {
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (globalTid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (globalTid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (globalTid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (globalTid % numWarpX + 1) * WMMA_N;

    __shared__ UIN rowIndexShared[SHARED_MEMORY_SIZE];
    __shared__ UIN colIndexShared[SHARED_MEMORY_SIZE];

    const int beginIdxInShared = threadIdx.x * NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
    const int endIdxInShared = beginIdxInShared + NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;

    const UIN beginIdx = matrixTileMappedToWarpIndex[globalTid];

    int count = 0;
    for (int loop = 0; loop < nnz; loop += SHARED_MEMORY_SIZE) {

        for (int mtxIdx = loop + beginIdxInShared, sharedIdx = beginIdxInShared;
             mtxIdx < nnz && sharedIdx < endIdxInShared;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll
        for (int sharedIdx = 0; sharedIdx < SHARED_MEMORY_SIZE && sharedIdx < sharedLoopEnd; ++sharedIdx) {
            const UIN curRow = rowIndexShared[sharedIdx];
            const UIN curCol = colIndexShared[sharedIdx];
            if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
                curCol >= colBeginOfTile && curCol < colEndOfTile) {
                matrixTileMappedToWarpIndexData[beginIdx + count] = loop + sharedIdx;
                ++count;
            }
        }
    }
}

template<typename OP>
__global__ void getIndexPerWarp_3(const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {

    const UIN warpIdInSDDMM = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (warpIdInSDDMM / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (warpIdInSDDMM / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (warpIdInSDDMM % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (warpIdInSDDMM % numWarpX + 1) * WMMA_N;

    __shared__ UIN rowIndexShared[SHARED_MEMORY_SIZE];
    __shared__ UIN colIndexShared[SHARED_MEMORY_SIZE];

    const int beginIdxOfSharedMemoryInThisThread = threadIdx.x * NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
    const int endIdxOfSharedMemoryInThisThread =
        beginIdxOfSharedMemoryInThisThread + NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
//    printf(" beginIdxOfSharedMemoryInThisThread = %d, endIdxOfSharedMemoryInThisThread = %d\n", beginIdxOfSharedMemoryInThisThread, endIdxOfSharedMemoryInThisThread);

    const UIN sparseMatrixDataInThisBlock = (blockIdx.y * SHARED_MEMORY_SIZE);

    // The amount of data in the last segment
    const UIN sharedLoopEnd = nnz - sparseMatrixDataInThisBlock;

#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
    for (UIN mtxIdx = sparseMatrixDataInThisBlock + beginIdxOfSharedMemoryInThisThread,
             sharedIdx = beginIdxOfSharedMemoryInThisThread;
         sharedIdx < endIdxOfSharedMemoryInThisThread && sharedIdx < sharedLoopEnd;
         ++sharedIdx, ++mtxIdx) {
        rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
        colIndexShared[sharedIdx] = colIndex[mtxIdx];
    }
    __syncthreads();

    op.init(gridDim, blockIdx, blockDim, threadIdx);
#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
    for (int sharedIdx = 0; sharedIdx < SHARED_MEMORY_SIZE && sharedIdx < sharedLoopEnd; ++sharedIdx) {
        const UIN curRow = rowIndexShared[sharedIdx];
        const UIN curCol = colIndexShared[sharedIdx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            op.cycle(sparseMatrixDataInThisBlock + sharedIdx);
        }
    }
    op.done();
}

template __global__ void getIndexPerWarp_3<updateScatteredNumOfIndexOperator_3>(const UIN numWarpX,
                                                                                const UIN nnz,
                                                                                const UIN *rowIndex,
                                                                                const UIN *colIndex,
                                                                                updateScatteredNumOfIndexOperator_3 op);
template __global__ void getIndexPerWarp_3<updateScatteredIndexDataPerWarpOperator_3>(const UIN numWarpX,
                                                                                      const UIN nnz,
                                                                                      const UIN *rowIndex,
                                                                                      const UIN *colIndex,
                                                                                      updateScatteredIndexDataPerWarpOperator_3 op);

__global__ void mergeScatteredNumOfIndex(const UIN numWarpsInSDDMM,
                                         const UIN numNNZBlocks,
                                         const UIN *scatteredNumOfIndex,
                                         UIN *mergedNumOfIndex) {
    const UIN globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTid >= numWarpsInSDDMM) {
        return;
    }
    const UIN numOfStoragePerBlockInOldData = numNNZBlocks * NUMBER_OF_THREADS_PER_BLOCK;
    const UIN beginIdxInThisThread = numOfStoragePerBlockInOldData * blockIdx.x + threadIdx.x;
    const UIN endIdxInThisBlock = numOfStoragePerBlockInOldData * (blockIdx.x + 1);
    const UIN numAddPerLoop = NUMBER_OF_THREADS_PER_BLOCK;

    UIN sum = 0;
    for (int idx = beginIdxInThisThread; idx < endIdxInThisBlock; idx += numAddPerLoop) {
        sum += scatteredNumOfIndex[idx];
    }

    mergedNumOfIndex[globalTid] = sum;
}

__global__ void sortScatteredIndexData(const UIN numWarpsInSDDMM,
                                       const UIN numNNZBlocks,
                                       const UIN *indexForNumOfIndex,
                                       const UIN *indexForScatteredNumOfIndex,
                                       const UIN *ScatteredIndexData,
                                       UIN *sortedIndexData) {
    const UIN globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTid >= numWarpsInSDDMM) {
        return;
    }

    const UIN beginIdxInThisThreadForStorage = indexForNumOfIndex[globalTid];

    const UIN numOfStoragePerBlockInOldData = numNNZBlocks * NUMBER_OF_THREADS_PER_BLOCK;
    const UIN beginIdxInThisThread = numOfStoragePerBlockInOldData * blockIdx.x + threadIdx.x;
    const UIN endIdxInThisBlock = numOfStoragePerBlockInOldData * (blockIdx.x + 1);
    const UIN numAddPerLoop = NUMBER_OF_THREADS_PER_BLOCK;

    UIN count = 0;
    for (int idx = beginIdxInThisThread; idx < endIdxInThisBlock; idx += numAddPerLoop) {
        for (int idxOfIndexForScatteredNumOfIndex = indexForScatteredNumOfIndex[idx];
             idxOfIndexForScatteredNumOfIndex < indexForScatteredNumOfIndex[idx + 1];
             ++idxOfIndexForScatteredNumOfIndex) {
            sortedIndexData[beginIdxInThisThreadForStorage + count] =
                ScatteredIndexData[idxOfIndexForScatteredNumOfIndex];
            ++count;
        }
    }
}