#pragma once

#include <cstdint>
#include <cuda_fp16.h>

using UIN = uint32_t;

// The dimension supported by WMMA
#define WMMA_16_16_16
//#define WMMA_32_8_16
//#define WMMA_8_32_16

#ifdef WMMA_16_16_16
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#endif // WMMA_16_16_16

#ifdef WMMA_32_8_16
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#endif // WMMA_32_8_16

#ifdef WMMA_8_32_16
const int WMMA_M = 8;
const int WMMA_N = 32;
const int WMMA_K = 16;
#endif // WMMA_8_32_16

using MATRIX_A_TYPE = __half;
using MATRIX_B_TYPE = __half;
using MATRIX_C_TYPE = float;

const int WARP_SIZE = 32;

const int NUM_OF_WARP_X_PER_BLOCK = 4;
const int NUM_OF_Y_PER_BLOCK = 4;

const int MATRIX_A_TILE_SIZE_PER_BLOCK = WMMA_M * NUM_OF_Y_PER_BLOCK * WMMA_K * NUM_OF_WARP_X_PER_BLOCK;
const int MATRIX_B_TILE_SIZE_PER_BLOCK = WMMA_K * NUM_OF_Y_PER_BLOCK * WMMA_N * NUM_OF_WARP_X_PER_BLOCK;

const int MATRIX_A_TILE_LEADING_DIMENSION = WMMA_K * NUM_OF_WARP_X_PER_BLOCK;
const int MATRIX_B_TILE_LEADING_DIMENSION = WMMA_N * NUM_OF_WARP_X_PER_BLOCK;

const int MEMORY_ACCESS_PER_THREAD =
    MATRIX_A_TILE_SIZE_PER_BLOCK / (NUM_OF_Y_PER_BLOCK * NUM_OF_WARP_X_PER_BLOCK * WARP_SIZE);

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
      blockDim_.x = NUM_OF_WARP_X_PER_BLOCK * WARP_SIZE;
      blockDim_.y = NUM_OF_Y_PER_BLOCK;

      const int numCountColOfOutputMatrixPerBlock = WMMA_N * blockDim_.x / WARP_SIZE;
      const int numCountRowOfOutputMatrixPerBlock = WMMA_M * blockDim_.y;
      gridDim_.x = (NForTensorCore(N) + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
      gridDim_.y = (MForTensorCore(M) + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;

      warpOrder_ = warpOrder;

      numWarpX_ = gridDim_.x * blockDim_.x / WARP_SIZE;
      numWarpY_ = gridDim_.y * blockDim_.y;;
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

  inline dim3 gridDim() const {
      return gridDim_;
  }

  inline dim3 blockDim() const {
      return blockDim_;
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

      globalWarpId_ = (globalThreadIdxX_ / WARP_SIZE) + globalThreadIdxY_ * (gridDim_.x * blockDim_.x / WARP_SIZE);

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

  dim3 gridDim_;
  dim3 blockDim_;

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
#ifdef WMMA_16_16_16
    positionCalculator_m16n16k16(tileRow, tileCol, row, col, fragmentInformation);
#endif //WMMA_16_16_16

#ifdef WMMA_32_16_16
    positionCalculator_m32n8k16(tileRow, tileCol, row, col, fragmentInformation);
#endif //WMMA_32_16_16

#ifdef WMMA_8_32_16
    positionCalculator_m8n32k16(tileRow, tileCol, row, col, fragmentInformation);
#endif //WMMA_8_32_16
}