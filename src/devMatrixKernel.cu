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
//    extern __shared__ UIN rowIndexShared[];
//    const int oneThread = 5000000 / 1024;
//    for (int i = threadIdx.x * oneThread; i < (threadIdx.x + 1) * oneThread && i < 10000; ++i) {
//        rowIndexShared[i] = rowIndex[i];
//    }
//    __syncthreads();

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

//    for (int idx = 10000; idx < nnz; ++idx) {
//        const UIN curRow = rowIndex[idx];
//        const UIN curCol = colIndex[idx];
//        if (curRow >= rowBeginOfTile && curRow < rowEndOfTile &&
//            curCol >= colBeginOfTile && curCol < colEndOfTile) {
//            ++num;
//        }
//    }
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