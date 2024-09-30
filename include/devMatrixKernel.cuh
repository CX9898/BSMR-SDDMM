#pragma once

#include "TensorCoreConfig.cuh"

//const UIN NUM_READ_DATA_BY_ONE_THREAD = 8;
//const UIN NUM_THREAD_PER_BLOCK = 512;
//const UIN NUM_READ_DATA_BY_ONE_BLOCK = NUM_READ_DATA_BY_ONE_THREAD * NUM_THREAD_PER_BLOCK;

const int NumberOfOperationsOnSharedByOneThread = 8;
const int NumberOfThreadsPerBlock = 512;
const int SharedMemorySize = NumberOfOperationsOnSharedByOneThread * NumberOfThreadsPerBlock;

template<typename T>
__global__ void getValuesFromDenseData(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                       const UIN *rowIndex, const UIN *colIndex,
                                       const T *denseData, T *output);

__global__ void getNumIndexPerWarp_1(const UIN size, const UIN numWarpX,
                                     const UIN numTileM, const UIN numTileN,
                                     const UIN nnz,
                                     const UIN *rowIndex,
                                     const UIN *colIndex,
                                     UIN *numIndexPerWarp);

__global__ void getTileIndexDataPerWarp(const UIN size, const UIN numWarpX,
                                        const UIN numTileM, const UIN numTileN,
                                        const UIN nnz,
                                        const UIN *rowIndex,
                                        const UIN *colIndex,
                                        const UIN *matrixTileMappedToWarpIndex,
                                        UIN *matrixTileMappedToWarpIndexData);

__global__ void getNumIndexPerWarp_2(const UIN size, const UIN numWarpX,
                                     const UIN numTileM, const UIN numTileN,
                                     const UIN nnz,
                                     const UIN *rowIndex,
                                     const UIN *colIndex,
                                     UIN *numIndexPerWarp);

__global__ void getTileIndexDataPerWarp_2(const UIN size, const UIN numWarpX,
                                          const UIN numTileM, const UIN numTileN,
                                          const UIN nnz,
                                          const UIN *rowIndex,
                                          const UIN *colIndex,
                                          const UIN *matrixTileMappedToWarpIndex,
                                          UIN *matrixTileMappedToWarpIndexData);

class updateNumIndexPerWarp {
 public:
  updateNumIndexPerWarp(UIN *data) : nums_(data) {}
  inline __device__ void init() {

  }
  inline __device__ void cycle() {
      ++num_;
  }
  inline __device__ void done(size_t idx) {
      nums_[idx] = num_;
  }

 private:
  UIN num_ = 0;

  UIN *nums_;
};

class updateIndexDataPerWarp {
 public:
  updateIndexDataPerWarp() {

  }
  inline __device__ void init() {

  }
  inline __device__ void cycle() {

  }
  inline __device__ void done(size_t idx) {

  }

 private:
  UIN dataIdx_ = 0;

  UIN *indexData_;
};

template<typename OP>
__global__ void getIndexPerWarp_3(const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op);