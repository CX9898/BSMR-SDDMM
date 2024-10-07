#pragma once

#include "TensorCoreConfig.cuh"

const int NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD = 8;
const int NUMBER_OF_THREADS_PER_BLOCK = 512;
const int SHARED_MEMORY_SIZE = NUMBER_OF_OPERATIONS_ON_SHARED_MEMORY_BY_ONE_THREAD * NUMBER_OF_THREADS_PER_BLOCK;

template<typename T>
__global__ void getValuesFromDenseData(const UIN row, const UIN col, const UIN nnz, const UIN ld,
                                       const UIN *rowIndex, const UIN *colIndex,
                                       const T *denseData, T *output);

class updateNumOfIndexOperator_1 {
 public:
  updateNumOfIndexOperator_1(UIN *nums) : nums_(nums) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      idxInThisThread = _blockIdx.x * _blockDim.x + _threadIdx.x;
  }
  inline __device__ void cycle(UIN mtxIdx) {
      ++num_;
  }
  inline __device__ void done() {
      nums_[idxInThisThread] = num_;
  }

 private:
  UIN idxInThisThread;
  UIN num_ = 0;

  UIN *nums_;
};

class updateIndexDataPerWarpOperator_1 {
 public:
  updateIndexDataPerWarpOperator_1(const UIN *indexForNumOfIndex, UIN *indexData) :
      indexForNumOfIndex_(indexForNumOfIndex), indexData_(indexData) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      UIN idxInThisThread = _blockIdx.x * _blockDim.x + _threadIdx.x;
      indexOfStartStoringInThisThread_ = indexForNumOfIndex_[idxInThisThread];
  }
  inline __device__ void cycle(UIN mtxIdx) {
      indexData_[indexOfStartStoringInThisThread_ + count_] = mtxIdx;
      ++count_;
  }
  inline __device__ void done() {}

 private:
  UIN count_ = 0;
  UIN indexOfStartStoringInThisThread_;

  const UIN *indexForNumOfIndex_;
  UIN *indexData_;
};

template<typename OP>
__global__ void getIndexPerWarp_1(const UIN size, const UIN numWarpX,
                                     const UIN numTileM, const UIN numTileN,
                                     const UIN nnz,
                                     const UIN *rowIndex,
                                     const UIN *colIndex,
                                     OP op);

class updateNumOfIndexOperator_2 {
 public:
  updateNumOfIndexOperator_2(UIN *nums) : nums_(nums) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      idxInThisThread = _blockIdx.x * _blockDim.x + _threadIdx.x;
  }
  inline __device__ void cycle(UIN mtxIdx) {
      ++num_;
  }
  inline __device__ void done() {
      nums_[idxInThisThread] = num_;
  }

 private:
  UIN idxInThisThread;
  UIN num_ = 0;

  UIN *nums_;
};

class updateIndexDataPerWarpOperator_2 {
 public:
  updateIndexDataPerWarpOperator_2(const UIN *indexForNumOfIndex, UIN *indexData) :
      indexForNumOfIndex_(indexForNumOfIndex), indexData_(indexData) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      UIN idxInThisThread = _blockIdx.x * _blockDim.x + _threadIdx.x;
      indexOfStartStoringInThisThread_ = indexForNumOfIndex_[idxInThisThread];
  }
  inline __device__ void cycle(UIN mtxIdx) {
      indexData_[indexOfStartStoringInThisThread_ + count_] = mtxIdx;
      ++count_;
  }
  inline __device__ void done() {}

 private:
  UIN count_ = 0;
  UIN indexOfStartStoringInThisThread_;

  const UIN *indexForNumOfIndex_;
  UIN *indexData_;
};

template<typename OP>
__global__ void getIndexPerWarp_2(const UIN size, const UIN numWarpX,
                                  const UIN numTileM, const UIN numTileN,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op);

class updateScatteredNumOfIndexOperator_3 {
 public:
  updateScatteredNumOfIndexOperator_3(UIN *nums) : nums_(nums) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      idxInThisThread = _gridDim.y * _blockDim.x * _blockIdx.x + _blockIdx.y * _blockDim.x + _threadIdx.x;
  }
  inline __device__ void cycle(UIN mtxIdx) {
      ++num_;
  }
  inline __device__ void done() {
      nums_[idxInThisThread] = num_;
  }

 private:
  UIN idxInThisThread;
  UIN num_ = 0;

  UIN *nums_;
};

class updateScatteredIndexDataPerWarpOperator_3 {
 public:
  updateScatteredIndexDataPerWarpOperator_3(const UIN *indexForScatteredNumOfIndex, UIN *scatteredIndexData) :
      indexForScatteredNumOfIndex_(indexForScatteredNumOfIndex), scatteredIndexData_(scatteredIndexData) {}

  inline __device__ void init(dim3 _gridDim, dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      UIN idxInThisThread = _gridDim.y * _blockDim.x * _blockIdx.x + _blockIdx.y * _blockDim.x + _threadIdx.x;
      indexOfStartStoringInThisThread_ = indexForScatteredNumOfIndex_[idxInThisThread];
  }
  inline __device__ void cycle(UIN mtxIdx) {
      scatteredIndexData_[indexOfStartStoringInThisThread_ + count_] = mtxIdx;
      ++count_;
  }
  inline __device__ void done() {}

 private:
  UIN count_ = 0;
  UIN indexOfStartStoringInThisThread_;

  const UIN *indexForScatteredNumOfIndex_;
  UIN *scatteredIndexData_;
};

/**
 * grid uses two dimensions :
 *  The grid X-axis is used to iteratively compute the 'warp',
 *  The grid Y-axis is used to iteratively compute 'nnz'
 * block uses one dimensions : NUMBER_OF_THREADS_PER_BLOCK
 **/
template<typename OP>
__global__ void getIndexPerWarp_3(const UIN numWarpX,
                                  const UIN nnz,
                                  const UIN *rowIndex,
                                  const UIN *colIndex,
                                  OP op);

__global__ void mergeScatteredNumOfIndex(const UIN numWarpsInSDDMM,
                                         const UIN numNNZBlocks,
                                         const UIN *scatteredNumOfIndex,
                                         UIN *mergedNumOfIndex);

__global__ void sortScatteredIndexData(const UIN numWarpsInSDDMM,
                                       const UIN numNNZBlocks,
                                       const UIN *indexForNumOfIndex,
                                       const UIN *indexForScatteredNumOfIndex,
                                       const UIN *ScatteredIndexData,
                                       UIN *sortedIndexData);