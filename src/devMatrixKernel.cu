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

    const UIN beginIdx = matrixTileMappedToWarpIndex[tid];

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

__global__ void getNumIndexPerWarp_2(const UIN size, const UIN numWarpX,
                                     const UIN numTileM, const UIN numTileN,
                                     const UIN nnz,
                                     const UIN *rowIndex,
                                     const UIN *colIndex,
                                     UIN *numIndexPerWarp) {
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

    const UIN rowBeginOfTile = (globalTid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (globalTid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (globalTid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (globalTid % numWarpX + 1) * WMMA_N;

    __shared__ UIN rowIndexShared[SharedMemorySize];
    __shared__ UIN colIndexShared[SharedMemorySize];

    const int sharedBeginIdx = threadIdx.x * NumberOfOperationsOnSharedByOneThread;
    const int sharedEndIdx = sharedBeginIdx + NumberOfOperationsOnSharedByOneThread;
//    printf(" sharedBeginIdx = %d, sharedEndIdx = %d\n", sharedBeginIdx, sharedEndIdx);

    UIN num = 0;
    for (int loop = 0; loop < nnz; loop += SharedMemorySize) {

#pragma unroll
        for (int mtxIdx = loop + sharedBeginIdx, sharedIdx = sharedBeginIdx;
             mtxIdx < nnz && sharedIdx < sharedEndIdx;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll
        for (int sharedIdx = 0; sharedIdx < SharedMemorySize && sharedIdx < sharedLoopEnd; ++sharedIdx) {
            const UIN curRow = rowIndexShared[sharedIdx];
            const UIN curCol = colIndexShared[sharedIdx];
            if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
                curCol >= colBeginOfTile && curCol < colEndOfTile) {
                ++num;
            }
        }
    }
    numIndexPerWarp[globalTid] = num;
}

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

    __shared__ UIN rowIndexShared[SharedMemorySize];
    __shared__ UIN colIndexShared[SharedMemorySize];

    const int beginIdxInShared = threadIdx.x * NumberOfOperationsOnSharedByOneThread;
    const int endIdxInShared = beginIdxInShared + NumberOfOperationsOnSharedByOneThread;

    const UIN beginIdx = matrixTileMappedToWarpIndex[globalTid];

    int count = 0;
    for (int loop = 0; loop < nnz; loop += SharedMemorySize) {

        for (int mtxIdx = loop + beginIdxInShared, sharedIdx = beginIdxInShared;
             mtxIdx < nnz && sharedIdx < endIdxInShared;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll
        for (int sharedIdx = 0; sharedIdx < SharedMemorySize && sharedIdx < sharedLoopEnd; ++sharedIdx) {
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

    __shared__ UIN rowIndexShared[SharedMemorySize];
    __shared__ UIN colIndexShared[SharedMemorySize];

    const int sharedMemoryBeginIdxInThisThread = threadIdx.x * NumberOfOperationsOnSharedByOneThread;
    const int sharedMemoryEndIdxInThisThread = sharedMemoryBeginIdxInThisThread + NumberOfOperationsOnSharedByOneThread;
//    printf(" sharedMemoryBeginIdxInThisThread = %d, sharedMemoryEndIdxInThisThread = %d\n", sharedMemoryBeginIdxInThisThread, sharedMemoryEndIdxInThisThread);

    const UIN sparseMatrixDataInThisBlock = (blockIdx.y * SharedMemorySize);

    // The amount of data in the last segment
    const UIN sharedLoopEnd = nnz - sparseMatrixDataInThisBlock;

#pragma unroll NumberOfOperationsOnSharedByOneThread
    for (UIN mtxIdx = sparseMatrixDataInThisBlock + sharedMemoryBeginIdxInThisThread,
             sharedIdx = sharedMemoryBeginIdxInThisThread;
         sharedIdx < sharedMemoryEndIdxInThisThread && sharedIdx < sharedLoopEnd;
         ++sharedIdx, ++mtxIdx) {
        rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
        colIndexShared[sharedIdx] = colIndex[mtxIdx];
    }
    __syncthreads();

    op.init();
#pragma unroll NumberOfOperationsOnSharedByOneThread
    for (int sharedIdx = 0; sharedIdx < SharedMemorySize && sharedIdx < sharedLoopEnd; ++sharedIdx) {
        const UIN curRow = rowIndexShared[sharedIdx];
        const UIN curCol = colIndexShared[sharedIdx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            op.cycle();
        }
    }

    const UIN idxInThisThread =
        (gridDim.y * NumberOfThreadsPerBlock) * blockIdx.x + blockIdx.y * NumberOfThreadsPerBlock + threadIdx.x;
    op.done(idxInThisThread);
}

template __global__ void getIndexPerWarp_3<updateNumIndexPerWarp>(const UIN numWarpX,
                                                                  const UIN nnz,
                                                                  const UIN *rowIndex,
                                                                  const UIN *colIndex,
                                                                  updateNumIndexPerWarp op);
template __global__ void getIndexPerWarp_3<updateIndexDataPerWarp>(const UIN numWarpX,
                                                                   const UIN nnz,
                                                                   const UIN *rowIndex,
                                                                   const UIN *colIndex,
                                                                   updateIndexDataPerWarp op);

__global__ void mergeNumOfIndexPerWarp(const UIN numNNZBlocks,
                                       const UIN numWarpsInSDDMM,
                                       const UIN *numsOld,
                                       UIN *numsNew) {
    const UIN globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTid >= numWarpsInSDDMM) {
        return;
    }
    const UIN numOfStoragePerBlockInOldData = numNNZBlocks * NumberOfThreadsPerBlock;
    const UIN beginIdxInThisThread = numOfStoragePerBlockInOldData * blockIdx.x + threadIdx.x;
    const UIN endIdxInThisBlock = numOfStoragePerBlockInOldData * (blockIdx.x + 1);
    const UIN numAddPerLoop = NumberOfThreadsPerBlock;

    UIN sum = 0;
    for (int idx = beginIdxInThisThread; idx < endIdxInThisBlock; idx += numAddPerLoop) {
        sum += numsOld[idx];
    }

    numsNew[globalTid] = sum;
}

__global__ void getTileIndexDataPerWarp_3(const UIN size, const UIN numWarpX,
                                          const UIN numTileM, const UIN numTileN,
                                          const UIN nnz,
                                          const UIN *rowIndex,
                                          const UIN *colIndex,
                                          const UIN *matrixTileMappedToWarpIndex,
                                          UIN *matrixTileMappedToWarpIndexData) {
    const int globalBid = blockDim.x * blockIdx.x;
    const int globalTid = globalBid + threadIdx.x;

    const int curWarpX = globalTid % numWarpX;
    const int curWarpY = globalTid / numWarpX;

    const UIN rowBeginOfTile = (globalTid / numWarpX) * WMMA_M;
    const UIN rowEndOfTile = (globalTid / numWarpX + 1) * WMMA_M;
    const UIN colBeginOfTile = (globalTid % numWarpX) * WMMA_N;
    const UIN colEndOfTile = (globalTid % numWarpX + 1) * WMMA_N;

    __shared__ UIN rowIndexShared[SharedMemorySize];
    __shared__ UIN colIndexShared[SharedMemorySize];

    const int beginIdxInShared = threadIdx.x * NumberOfOperationsOnSharedByOneThread;
    const int endIdxInShared = beginIdxInShared + NumberOfOperationsOnSharedByOneThread;

    const UIN beginIdx = matrixTileMappedToWarpIndex[globalTid];

    int count = 0;
    for (int loop = 0; loop < nnz; loop += SharedMemorySize) {

        for (int mtxIdx = loop + beginIdxInShared, sharedIdx = beginIdxInShared;
             mtxIdx < nnz && sharedIdx < endIdxInShared;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        const UIN sharedLoopEnd = nnz - loop;

#pragma unroll
        for (int sharedIdx = 0; sharedIdx < SharedMemorySize && sharedIdx < sharedLoopEnd; ++sharedIdx) {
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