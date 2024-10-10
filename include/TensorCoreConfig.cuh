#pragma once

#include <cstdint>
#include <cuda_fp16.h>

using UIN = uint32_t;

using MATRIX_A_TYPE = __half;
using MATRIX_B_TYPE = __half;
using MATRIX_C_TYPE = float;

// The dimension supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_SIZE = 32;

const int NUM_OF_WARP_X_PER_BLOCK = 4;
const int NUM_OF_Y_PER_BLOCK = 4;

enum WarpOrder {
  x_major,
  y_major
};

/**
 * Used to record where elements in a matrix tile are stored in the cuda::wmma::fragment class
 **/
struct FragmentInformation {
  int laneId = -1;
  int index = -1;
};



/**
 * Configuration class for matrix multiplication using Tensor core
 **/
// TODO: Adjust according to WarpOrder
class TensorCoreConfig {
 public:
  TensorCoreConfig() = delete;

  TensorCoreConfig(size_t matrixRow, size_t matrixCol) {
      // TODO: Duplicate code, unsafe
      const int rowComplement = matrixRow % WMMA_M == 0 ? 0 : WMMA_M - matrixRow % WMMA_M;
      const int colComplement = matrixCol % WMMA_N == 0 ? 0 : WMMA_N - matrixCol % WMMA_N;

      const size_t matrixRowForTensorCore = matrixRow + rowComplement;
      const size_t matrixColForTensorCore = matrixCol + colComplement;

      block_.x = NUM_OF_WARP_X_PER_BLOCK * WARP_SIZE;
      block_.y = NUM_OF_Y_PER_BLOCK;

      const int numCountColOfOutputMatrixPerBlock = WMMA_M * block_.x / WARP_SIZE;
      const int numCountRowOfOutputMatrixPerBlock = WMMA_N * block_.y;
      grid_.x = (matrixColForTensorCore + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
      grid_.y = (matrixRowForTensorCore + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
  }

  dim3 grid() const {
      return grid_;
  }

  dim3 block() const {
      return block_;
  }

  size_t numWarpX() const {
      return grid_.x * block_.x / WARP_SIZE;
  }

  size_t numWarpY() const {
      return grid_.y * block_.y;
  }

  __device__ void initByKernel(dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      globalThreadIdxX_ = _blockDim.x * _blockIdx.x + _threadIdx.x;
      globalThreadIdxY_ = _blockDim.y * _blockIdx.y + _threadIdx.y;

      globalWarpId_ = (globalThreadIdxX_ / WARP_SIZE) + globalThreadIdxY_ * (grid_.x * block_.x / WARP_SIZE);

      laneId_ = globalThreadIdxX_ % WARP_SIZE;
  }

  __device__ size_t globalThreadIdxX() const {
      return globalThreadIdxX_;
  }
  __device__ size_t globalThreadIdxY() const {
      return globalThreadIdxY_;
  }
  __device__ size_t globalWarpId() const {
      return globalWarpId_;
  }
  __device__ int laneId() const {
      return laneId_;
  }
  __device__ size_t rowBeginOfTile() const {
      return globalThreadIdxY_ * WMMA_M;
  }
  __device__ size_t colBeginOfTile() const {
      return globalThreadIdxX_ / WARP_SIZE * WMMA_N;
  }
  __device__ size_t rowEndOfTile() const {
      return globalThreadIdxY_ * WMMA_M + WMMA_M;
  }
  __device__ size_t colEndOfTile() const {
      return globalThreadIdxX_ / WARP_SIZE * WMMA_N + WMMA_N;
  }
  inline __device__ void positionCalculator(const UIN tileRow, const UIN tileCol,
                                            const UIN row, const UIN col,
                                            FragmentInformation &fragmentInformation);

 private:
  dim3 grid_;
  dim3 block_;

  // kernel
  size_t globalThreadIdxX_;
  size_t globalThreadIdxY_;
  int globalWarpId_;
  int laneId_;
};

inline __device__ void positionCalculator_m16n16k16(const UIN tileRow, const UIN tileCol,
                                                    const UIN row, const UIN col,
                                                    FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        fragmentInformation.laneId = -1;
        fragmentInformation.index = -1;
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localRow % 8 * 4;
    const int isBigRow = localRow / 8;
    const int isBigCol = localCol / 8;
    fragmentInformation.laneId = beginLane + localCol % 8 / 2;
    fragmentInformation.index = isBigRow * 2 + isBigCol * 4 + localCol % 2;
}
inline __device__ void positionCalculator_m32n8k16(const UIN tileRow, const UIN tileCol,
                                                   const UIN row, const UIN col,
                                                   FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        fragmentInformation.laneId = -1;
        fragmentInformation.index = -1;
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

}
inline __device__ void positionCalculator_m8n32k16(const UIN tileRow, const UIN tileCol,
                                                   const UIN row, const UIN col,
                                                   FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        fragmentInformation.laneId = -1;
        fragmentInformation.index = -1;
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

}

inline __device__ void TensorCoreConfig::positionCalculator(const UIN tileRow, const UIN tileCol,
                                                            const UIN row, const UIN col,
                                                            FragmentInformation &fragmentInformation) {
    if (WMMA_M == 16 && WMMA_N == 16 && WMMA_K == 16) {
        positionCalculator_m16n16k16(tileRow, tileCol, row, col, fragmentInformation);
    }
    if (WMMA_M == 32 && WMMA_N == 8 && WMMA_K == 16) {
        positionCalculator_m32n8k16(tileRow, tileCol, row, col, fragmentInformation);
    }
    if (WMMA_M == 8 && WMMA_N == 32 && WMMA_K == 16) {
        positionCalculator_m8n32k16(tileRow, tileCol, row, col, fragmentInformation);
    }
}