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

template<typename OP>
__global__ void getIndexPerWarp_1(const UIN numWarpsInSDDMM, const UIN numWarpXInSDDMM,
                                  const UIN numTileM, const UIN numTileN,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {
    const int warpIdInSDDMM = blockDim.x * blockIdx.x + threadIdx.x;
    if (warpIdInSDDMM >= numWarpsInSDDMM) {
        return;
    }

    const int curWarpX = warpIdInSDDMM % numWarpXInSDDMM;
    const int curWarpY = warpIdInSDDMM / numWarpXInSDDMM;
    if (curWarpX > numTileN || curWarpY > numTileM) {
        return;
    }

    const UIN rowBeginOfTile = (warpIdInSDDMM / numWarpXInSDDMM) * WMMA_M;
    const UIN rowEndOfTile = (warpIdInSDDMM / numWarpXInSDDMM + 1) * WMMA_M;
    const UIN colBeginOfTile = (warpIdInSDDMM % numWarpXInSDDMM) * WMMA_N;
    const UIN colEndOfTile = (warpIdInSDDMM % numWarpXInSDDMM + 1) * WMMA_N;

    op.init(gridDim, blockIdx, blockDim, threadIdx);
    for (int mtxIdx = 0; mtxIdx < nnz; ++mtxIdx) {
        const UIN curRow = rowIndex[mtxIdx];
        const UIN curCol = colIndex[mtxIdx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            op.cycle(mtxIdx);
        }
    }

    op.done();
}

template __global__ void getIndexPerWarp_1<updateNumOfIndexOperator_1>(const UIN size,
                                                                       const UIN numWarpX,
                                                                       const UIN numTileM,
                                                                       const UIN numTileN,
                                                                       const UIN nnz,
                                                                       const UIN *rowIndex,
                                                                       const UIN *colIndex,
                                                                       updateNumOfIndexOperator_1 op);
template __global__ void getIndexPerWarp_1<updateIndexDataPerWarpOperator_1>(const UIN size,
                                                                             const UIN numWarpX,
                                                                             const UIN numTileM,
                                                                             const UIN numTileN,
                                                                             const UIN nnz,
                                                                             const UIN *rowIndex,
                                                                             const UIN *colIndex,
                                                                             updateIndexDataPerWarpOperator_1 op);

/**
 * Pipeline method
 * error
 **/
template<typename OP>
__global__ void getIndexPerWarp_2(const UIN size, const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {

    __shared__ UIN rowIndexShared[SHARED_MEMORY_SIZE];
    __shared__ UIN colIndexShared[SHARED_MEMORY_SIZE];

    const int sharedBeginIdx = threadIdx.x * NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;
    const int sharedEndIdx = sharedBeginIdx + NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD;

    const int warpIdInSDDMM = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (warpIdInSDDMM / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (warpIdInSDDMM / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (warpIdInSDDMM % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (warpIdInSDDMM % numWarpX + 1) * WMMA_N;

    op.init(gridDim, blockIdx, blockDim, threadIdx);
    for (int loop = 0; loop < nnz; loop += SHARED_MEMORY_SIZE) {

#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
        for (int mtxIdx = loop + sharedBeginIdx, sharedIdx = sharedBeginIdx;
             mtxIdx < nnz && sharedIdx < sharedEndIdx;
             ++mtxIdx, ++sharedIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();
        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
        for (int sharedIdx = 0; sharedIdx < SHARED_MEMORY_SIZE && sharedIdx < sharedLoopEnd; ++sharedIdx) {
            const UIN curRow = rowIndexShared[sharedIdx];
            const UIN curCol = colIndexShared[sharedIdx];

            if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
                curCol >= colBeginOfTile && curCol < colEndOfTile) {
                op.cycle(loop + sharedIdx);
            }
        }

        __syncthreads();
    }
    op.done();
}

template __global__ void getIndexPerWarp_2<updateNumOfIndexOperator_2>(const UIN size,
                                                                       const UIN numWarpX,
                                                                       const UIN nnz,
                                                                       const UIN *rowIndex,
                                                                       const UIN *colIndex,
                                                                       updateNumOfIndexOperator_2 op);
template __global__ void getIndexPerWarp_2<updateIndexDataPerWarpOperator_2>(const UIN size,
                                                                             const UIN numWarpX,
                                                                             const UIN nnz,
                                                                             const UIN *rowIndex,
                                                                             const UIN *colIndex,
                                                                             updateIndexDataPerWarpOperator_2 op);

template<typename OP>
__global__ void getIndexPerWarp_3(const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {

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

    const UIN warpIdInSDDMM = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (warpIdInSDDMM / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (warpIdInSDDMM / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (warpIdInSDDMM % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (warpIdInSDDMM % numWarpX + 1) * WMMA_N;

    op.init(gridDim, blockIdx, blockDim, threadIdx);
#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
    for (int sharedIdx = 0; sharedIdx < SHARED_MEMORY_SIZE && sharedIdx < sharedLoopEnd; ++sharedIdx) {
        const UIN curRow = rowIndexShared[sharedIdx];
        const UIN curCol = colIndexShared[sharedIdx];
        // TODO : 每个线程的判断结果都不一样, 太多分支
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

__global__ void mergeScatteredNumOfIndex_3(const UIN numWarpsInSDDMM,
                                           const UIN numNNZBlocks,
                                           const UIN *scatteredNumOfIndex,
                                           UIN *mergedNumOfIndex) {
    const UIN warpIdInSDDMM = blockIdx.x * blockDim.x + threadIdx.x;
    if (warpIdInSDDMM >= numWarpsInSDDMM) {
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

    mergedNumOfIndex[warpIdInSDDMM] = sum;
}

__global__ void sortScatteredIndexData_3(const UIN numWarpsInSDDMM,
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

    const UIN numOfStoragePerYGridInOldData = numNNZBlocks * NUMBER_OF_THREADS_PER_BLOCK;
    const UIN beginIdxInThisThread = numOfStoragePerYGridInOldData * blockIdx.x + threadIdx.x;
    const UIN endIdxInThisBlock = numOfStoragePerYGridInOldData * (blockIdx.x + 1);
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

template<typename OP>
__global__ void getIndexPerWarp_4(TensorCoreConfig tensorCoreConfig,
                                  const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op) {

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

    const int localWarpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const UIN warpIdInSDDMM = blockIdx.x * NUMBER_OF_CALCULATED_BY_ONE_BLOCK + localWarpId;

    const UIN rowBeginOfTile = (warpIdInSDDMM / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (warpIdInSDDMM / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (warpIdInSDDMM % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (warpIdInSDDMM % numWarpX + 1) * WMMA_N;

    op.init(gridDim, blockIdx, blockDim, threadIdx);
#pragma unroll NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD
    for (int sharedIdx = laneId;
         sharedIdx < SHARED_MEMORY_SIZE && sharedIdx < sharedLoopEnd;
         sharedIdx += WARP_SIZE) {
        const UIN curRow = rowIndexShared[sharedIdx];
        const UIN curCol = colIndexShared[sharedIdx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            op.cycle(sparseMatrixDataInThisBlock + sharedIdx);
        }
    }
    op.done();
}

template __global__ void getIndexPerWarp_4<updateScatteredNumOfIndexOperator_4>(TensorCoreConfig tensorCoreConfig,
                                                                                const UIN numWarpX,
                                                                                const UIN nnz,
                                                                                const UIN *rowIndex,
                                                                                const UIN *colIndex,
                                                                                updateScatteredNumOfIndexOperator_4 op);
template __global__ void getIndexPerWarp_4<updateScatteredIndexDataPerWarpOperator_4>(TensorCoreConfig tensorCoreConfig,
                                                                                      const UIN numWarpX,
                                                                                      const UIN nnz,
                                                                                      const UIN *rowIndex,
                                                                                      const UIN *colIndex,
                                                                                      updateScatteredIndexDataPerWarpOperator_4 op);

__global__ void mergeScatteredNumOfIndex_4(const UIN numWarpsInSDDMM,
                                           const UIN numNNZBlocks,
                                           const UIN *scatteredNumOfIndex,
                                           UIN *mergedNumOfIndex) {

    const UIN warpIdInSDDMM = blockIdx.x * blockDim.x + threadIdx.x;
    if (warpIdInSDDMM >= numWarpsInSDDMM) {
        return;
    }

    const UIN numOfStoragePerYGridInOldData = NUMBER_OF_CALCULATED_BY_ONE_BLOCK * numNNZBlocks;
    const UIN blockIdxXInOldData = warpIdInSDDMM / NUMBER_OF_CALCULATED_BY_ONE_BLOCK;
    const UIN threadIdxInOldData = warpIdInSDDMM % NUMBER_OF_CALCULATED_BY_ONE_BLOCK;
    const UIN beginIdxInThisThread = numOfStoragePerYGridInOldData * blockIdxXInOldData + threadIdxInOldData;
    const UIN endIdxInThisBlock = numOfStoragePerYGridInOldData * (blockIdxXInOldData + 1);
    const UIN numAddPerLoop = NUMBER_OF_CALCULATED_BY_ONE_BLOCK;

    UIN sum = 0;
    for (int idx = beginIdxInThisThread; idx < endIdxInThisBlock; idx += numAddPerLoop) {
        sum += scatteredNumOfIndex[idx];
    }

    mergedNumOfIndex[warpIdInSDDMM] = sum;
}