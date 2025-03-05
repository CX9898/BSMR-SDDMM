#pragma once

#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using UIN = uint32_t;
constexpr UIN MAX_UIN = std::numeric_limits<UIN>::max();

// The dimension supported by WMMA
//#define WMMA_16_16_16
//#define WMMA_32_8_16
//#define WMMA_8_32_16
#define WMMA_16_16_8

#ifdef WMMA_16_16_16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
#endif // WMMA_16_16_16

#ifdef WMMA_32_8_16
constexpr int  WMMA_M = 32;
constexpr int  WMMA_N = 8;
constexpr int  WMMA_K = 16;
#endif // WMMA_32_8_16

#ifdef WMMA_8_32_16
constexpr int  WMMA_M = 8;
constexpr int  WMMA_N = 32;
constexpr int  WMMA_K = 16;
#endif // WMMA_8_32_16

#ifdef WMMA_16_16_8
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
#endif // WMMA_16_16_8

#if defined(WMMA_16_16_16) || defined(WMMA_32_8_16) || defined(WMMA_8_32_16)
using MATRIX_A_TYPE = __half;
using MATRIX_B_TYPE = __half;
using MATRIX_C_TYPE = float;

#ifdef __CUDACC__
using MATRIX_A_TYPE_FRAGMENT = __half;
using MATRIX_B_TYPE_FRAGMENT = __half;
#endif // __CUDACC__

#endif // defined(WMMA_16_16_16) || defined(WMMA_32_8_16) || defined(WMMA_8_32_16)

#ifdef WMMA_16_16_8
using MATRIX_A_TYPE = float;
using MATRIX_B_TYPE = float;
using MATRIX_C_TYPE = float;

#ifdef __CUDACC__
using MATRIX_A_TYPE_FRAGMENT = ::nvcuda::wmma::precision::tf32;
using MATRIX_B_TYPE_FRAGMENT = ::nvcuda::wmma::precision::tf32;
#endif // __CUDACC__

#endif // WMMA_16_16_8

constexpr int WARP_SIZE = 32;

inline __host__ __device__ void calculateFragmentLaneAndIndex_m16n16(const UIN tileRow, const UIN tileCol,
                                                                     const UIN row, const UIN col,
                                                                     UIN &laneId, UIN &indexOfFragment) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localRow % 8 * 4;
    const int isBigRow = localRow / 8;
    const int isBigCol = localCol / 8;
    laneId = beginLane + localCol % 8 / 2;
    indexOfFragment = isBigRow * 2 + isBigCol * 4 + localCol % 2;
}
inline __host__ __device__ void calculateFragmentLaneAndIndex_m32n8(const UIN tileRow, const UIN tileCol,
                                                                    const UIN row, const UIN col,
                                                                    UIN &laneId, UIN &indexOfFragment) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localRow % 8 * 4;
    const int groupId = localRow / 8;
    const int isColOdd = localCol % 2;
    laneId = beginLane + localCol / 2;
    indexOfFragment = groupId * 2 + isColOdd;
}
inline __host__ __device__ void calculateFragmentLaneAndIndex_m8n32(const UIN tileRow, const UIN tileCol,
                                                                    const UIN row, const UIN col,
                                                                    UIN &laneId, UIN &indexOfFragment) {
    if (tileRow > row || tileCol > col || tileRow + WMMA_M <= row || tileCol + WMMA_N <= col) {
        return;
    }
    const int localRow = row - tileRow;
    const int localCol = col - tileCol;

    const int beginLane = localCol % 8 * 4;
    const int groupId = localCol / 8;
    const int isColOdd = localRow % 2;
    laneId = beginLane + localRow / 2;
    indexOfFragment = groupId * 2 + isColOdd;
}

inline __host__ __device__ void calculateFragmentLaneAndIndex(const UIN tileRow, const UIN tileCol,
                                                              const UIN row, const UIN col,
                                                              UIN &laneId, UIN &indexOfFragment) {
#if defined(WMMA_16_16_16) || defined(WMMA_16_16_8)
    calculateFragmentLaneAndIndex_m16n16(tileRow, tileCol, row, col, laneId, indexOfFragment);
#endif //WMMA_16_16_16

#ifdef WMMA_32_16_16
    calculateFragmentLaneAndIndex_m32n8(tileRow, tileCol, row, col, laneId, indexOfFragment);
#endif //WMMA_32_16_16

#ifdef WMMA_8_32_16
    calculateFragmentLaneAndIndex_m8n32(tileRow, tileCol, row, col, laneId, indexOfFragment);
#endif //WMMA_8_32_16
}

inline __host__ __device__ void calculateFragmentCoordinates_m16n16(const UIN laneId, const UIN indexOfFragment,
                                                                    UIN &row, UIN &col) {
    // Divide the lanes into groups of 4
    const UIN laneGroupId = laneId >> 2; // laneId / 4
    const UIN localIdInLaneGroup = laneId & 3; // laneId % 4

    const UIN indexGroupId = indexOfFragment >> 1; // indexOfFragment / 2
    const UIN isOddIndex = indexOfFragment & 1; // indexOfFragment % 2
    const UIN isBigLaneGroup = indexOfFragment >> 2; // indexOfFragment / 4

//    const UIN isOddIndexGroupId = indexGroupId % 2

    row = laneGroupId + (indexGroupId & 1) * 8;
    // row = laneGroupId + 8 * isOddIndexGroupId;
    col = (localIdInLaneGroup << 1) + isOddIndex + (isBigLaneGroup << 3);
    // col = localIdInLaneGroup * 2 + isOddIndex + 8 * isBigLaneGroup;
}

inline __host__ __device__ void calculateFragmentCoordinates_m32n8(const UIN laneId, const UIN indexOfFragment,
                                                                   UIN &row, UIN &col) {
    // Divide the lanes into groups of 4
    const UIN laneGroupId = laneId >> 2;  // laneId / 4
    const UIN localIdInLaneGroup = laneId & 3;  // laneId % 4

    const UIN indexGroupId = indexOfFragment >> 1;  // indexOfFragment / 2
    const UIN isOddIndex = indexOfFragment & 1;  // indexOfFragment % 2

    row = (indexGroupId << 3) + laneGroupId;  // indexGroupId * 8 + laneGroupId
    col = (localIdInLaneGroup << 1) + isOddIndex;  // localIdInLaneGroup * 2 + isOddIndex
}

inline __host__ __device__ void calculateFragmentCoordinates_m8n32(const UIN laneId, const UIN indexOfFragment,
                                                                   UIN &row, UIN &col) {
    // Divide the lanes into groups of 4
    const UIN laneGroupId = laneId >> 2;  // laneId / 4
    const UIN localIdInLaneGroup = laneId & 3;  // laneId % 4

    const UIN indexGroupId = indexOfFragment >> 1;  // indexOfFragment / 2
    const UIN isOddIndex = indexOfFragment & 1;  // indexOfFragment % 2

    row = (localIdInLaneGroup << 1) + isOddIndex;  // localIdInLaneGroup * 2 + isOddIndex
    col = (indexGroupId << 3) + laneGroupId;  // indexGroupId * 8 + laneGroupId
}

inline __host__ __device__ void calculateFragmentCoordinates(const UIN laneId, const UIN indexOfFragment,
                                                             UIN &row, UIN &col) {
#if defined(WMMA_16_16_16) || defined(WMMA_16_16_8)
    calculateFragmentCoordinates_m16n16(laneId, indexOfFragment, row, col);
#endif //WMMA_16_16_16

#ifdef WMMA_32_16_16
    calculateFragmentCoordinates_m32n8(laneId, indexOfFragment, row, col);
#endif //WMMA_32_16_16

#ifdef WMMA_8_32_16
    calculateFragmentCoordinates_m8n32(laneId, indexOfFragment, row, col);
#endif //WMMA_8_32_16
}