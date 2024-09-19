#include "devMatrixKernel.cuh"

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

    const size_t rowBeginOfTile = (tid / numWarpX) * WMMA_M;
    const size_t rowEndOfTile = (tid / numWarpX + 1) * WMMA_M;
    const size_t colBeginOfTile = (tid % numWarpX) * WMMA_N;
    const size_t colEndOfTile = (tid % numWarpX + 1) * WMMA_N;

    UIN num = 0;
    for (int idx = 0; idx < nnz; ++idx) {
        const size_t curRow = rowIndex[idx];
        const size_t curCol = colIndex[idx];
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

    const size_t rowBeginOfTile = (tid / numWarpX) * WMMA_M;
    const size_t rowEndOfTile = (tid / numWarpX + 1) * WMMA_M;
    const size_t colBeginOfTile = (tid % numWarpX) * WMMA_N;
    const size_t colEndOfTile = (tid % numWarpX + 1) * WMMA_N;

    const UIN beginIdx = matrixTileMappedToWarpIndex[tid];

    UIN count = 0;
    for (int idx = 0; idx < nnz; ++idx) {
        const size_t curRow = rowIndex[idx];
        const size_t curCol = colIndex[idx];
        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
            curCol >= colBeginOfTile && curCol < colEndOfTile) {
            matrixTileMappedToWarpIndexData[beginIdx + count] = idx;
            ++count;
        }
    }

}