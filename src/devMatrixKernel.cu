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

__global__ void getNumIndexPerWarp(const UIN size, const UIN numWarpX,
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
    const int globalBid = blockDim.x * blockIdx.x;
    const int globalTid = globalBid + threadIdx.x;
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

    __shared__ UIN rowIndexShared[numOfMemoryReadInOneLoop];
    __shared__ UIN colIndexShared[numOfMemoryReadInOneLoop];

    const int beginIdxInShared = threadIdx.x * numOfMemoryReadInOneLoopOfOneThread;
    const int endIdxInShared = beginIdxInShared + numOfMemoryReadInOneLoopOfOneThread;

    UIN num = 0;
    for (int loop = 0; loop < nnz; loop += numOfMemoryReadInOneLoop) {

        for (int mtxIdx = loop + beginIdxInShared, sharedIdx = beginIdxInShared;
             mtxIdx < nnz && sharedIdx < endIdxInShared;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        for (int sharedIdx = 0; sharedIdx < numOfMemoryReadInOneLoop; ++sharedIdx) {
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
    const int globalBid = blockDim.x * blockIdx.x;
    const int globalTid = globalBid + threadIdx.x;
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

    __shared__ UIN rowIndexShared[numOfMemoryReadInOneLoop];
    __shared__ UIN colIndexShared[numOfMemoryReadInOneLoop];

    const int beginIdxInShared = threadIdx.x * numOfMemoryReadInOneLoopOfOneThread;
    const int endIdxInShared = beginIdxInShared + numOfMemoryReadInOneLoopOfOneThread;

    const UIN beginIdx = matrixTileMappedToWarpIndex[globalTid];

    int count = 0;
    for (int loop = 0; loop < nnz; loop += numOfMemoryReadInOneLoop) {

        for (int mtxIdx = loop + beginIdxInShared, sharedIdx = beginIdxInShared;
             mtxIdx < nnz && sharedIdx < endIdxInShared;
             ++sharedIdx, ++mtxIdx) {
            rowIndexShared[sharedIdx] = rowIndex[mtxIdx];
            colIndexShared[sharedIdx] = colIndex[mtxIdx];
        }
        __syncthreads();

        for (int sharedIdx = 0; sharedIdx < numOfMemoryReadInOneLoop; ++sharedIdx) {
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