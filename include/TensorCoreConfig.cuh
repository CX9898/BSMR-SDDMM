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
  int laneId_ = -1;
  int index_ = -1;
};



/**
 * Configuration class for matrix multiplication using Tensor core
 **/
// TODO: Adjust according to WarpOrder
class TensorCoreConfig {
 public:
  TensorCoreConfig() = delete;

  TensorCoreConfig(UIN M, UIN N, WarpOrder warpOrder = WarpOrder::y_major) {
      block_.x = NUM_OF_WARP_X_PER_BLOCK * WARP_SIZE;
      block_.y = NUM_OF_Y_PER_BLOCK;

      const int numCountColOfOutputMatrixPerBlock = WMMA_N * block_.x / WARP_SIZE;
      const int numCountRowOfOutputMatrixPerBlock = WMMA_M * block_.y;
      grid_.x = (MForTensorCore(M) + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
      grid_.y = (NForTensorCore(N) + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;

      warpOrder_ = warpOrder;

      numWarpX_ = grid_.x * block_.x / WARP_SIZE;
      numWarpY_ = grid_.y * block_.y;;
  }

  inline UIN MForTensorCore(UIN M) const {
      const int MComplement = M % WMMA_M == 0 ? 0 : WMMA_M - M % WMMA_M;
      return M + MComplement;
  }

  inline UIN NForTensorCore(UIN N) const {
      const int NComplement = N % WMMA_N == 0 ? 0 : WMMA_N - N % WMMA_N;
      return N + NComplement;
  }

  inline UIN KForTensorCore(UIN K) const {
      const int KComplement = K % WMMA_K == 0 ? 0 : WMMA_K - K % WMMA_K;
      return K + KComplement;
  }

  inline dim3 grid() const {
      return grid_;
  }

  inline dim3 block() const {
      return block_;
  }

  inline UIN numWarpX() const {
      return numWarpX_;
  }

  inline UIN numWarpY() const {
      return numWarpY_;
  }

  inline UIN calculateWarpId(UIN row, UIN col) const {
      return row / WMMA_M * numWarpX_ + col / WMMA_N;
  }

  inline __device__ void initByKernel(dim3 _blockIdx, dim3 _blockDim, dim3 _threadIdx) {
      globalThreadIdxX_ = _blockDim.x * _blockIdx.x + _threadIdx.x;
      globalThreadIdxY_ = _blockDim.y * _blockIdx.y + _threadIdx.y;

      globalWarpId_ = (globalThreadIdxX_ / WARP_SIZE) + globalThreadIdxY_ * (grid_.x * block_.x / WARP_SIZE);

      laneId_ = _threadIdx.x % WARP_SIZE;
  }

  inline __device__ UIN globalThreadIdxX() const {
      return globalThreadIdxX_;
  }
  inline __device__ UIN globalThreadIdxY() const {
      return globalThreadIdxY_;
  }
  inline __device__ UIN globalWarpId() const {
      return globalWarpId_;
  }
  inline __device__ int laneId() const {
      return laneId_;
  }
  inline __device__ UIN rowBeginOfTile() const {
      return globalThreadIdxY_ * WMMA_M;
  }
  inline __device__ UIN colBeginOfTile() const {
      return globalThreadIdxX_ / WARP_SIZE * WMMA_N;
  }
  inline __device__ UIN rowEndOfTile() const {
      return globalThreadIdxY_ * WMMA_M + WMMA_M;
  }
  inline __device__ UIN colEndOfTile() const {
      return globalThreadIdxX_ / WARP_SIZE * WMMA_N + WMMA_N;
  }
  inline __device__ void positionCalculator(const UIN tileRow, const UIN tileCol,
                                            const UIN row, const UIN col,
                                            FragmentInformation &fragmentInformation);

 private:
  WarpOrder warpOrder_;

  dim3 grid_;
  dim3 block_;

  UIN numWarpX_;
  UIN numWarpY_;

  // kernel
  UIN globalThreadIdxX_;
  UIN globalThreadIdxY_;
  int globalWarpId_;
  int laneId_;
};

//__device__ void positionCalculator_m16n16k16(const UIN tileRow, const UIN tileCol,
//                                   const UIN row, const UIN col,
//                                   int &laneId_, int &idx) {
//    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
//        laneId_ = -1;
//        idx = -1;
//        return;
//    }
//    const int localRow = row - tileRow;
//    const int localCol = col - tileCol;
//
//    const int numberOfIterations = localCol % 8;
//
//    const int startLane = (localRow % 8) * 4;
//    laneId_ = startLane + numberOfIterations / 2;
//
//    const int addNum = numberOfIterations % 2;
//    if (localCol < 8) { // idx : 0~3
//        if (localRow < 8) { //  idx : 0~1 || 4~5
//            idx = addNum;
//        } else { // idx : 2~3 || 6~7
//            idx = 2 + addNum;
//        }
//    } else { // idx : 4~7
//        if (localRow < 8) { //  idx : 0~1 || 4~5
//            idx = 4 + addNum;
//        } else { // idx : 2~3 || 6~7
//            idx = 6 + addNum;
//        }
//    }
//}

inline __device__ void positionCalculator_m16n16k16(const UIN tileRow, const UIN tileCol,
                                                    const UIN row, const UIN col,
                                                    FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localRow % 8 * 4;
    const int isBigRow = localRow / 8;
    const int isBigCol = localCol / 8;
    fragmentInformation.laneId_ = beginLane + localCol % 8 / 2;
    fragmentInformation.index_ = isBigRow * 2 + isBigCol * 4 + localCol % 2;
}
inline __device__ void positionCalculator_m32n8k16(const UIN tileRow, const UIN tileCol,
                                                   const UIN row, const UIN col,
                                                   FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localRow % 8 * 4;
    const int groupId = localRow / 8;
    const int isColOdd = localCol % 2;
    fragmentInformation.laneId_ = beginLane + localCol / 2;
    fragmentInformation.index_ = groupId * 2 + isColOdd;
}
inline __device__ void positionCalculator_m8n32k16(const UIN tileRow, const UIN tileCol,
                                                   const UIN row, const UIN col,
                                                   FragmentInformation &fragmentInformation) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localCol % 8 * 4;
    const int groupId = localCol / 8;
    const int isColOdd = localRow % 2;
    fragmentInformation.laneId_ = beginLane + localRow / 2;
    fragmentInformation.index_ = groupId * 2 + isColOdd;
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