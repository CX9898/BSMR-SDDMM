#pragma once

#include <cstdio>
#include <string>
#include <cuda_runtime.h>

namespace cuUtil {

template<typename T>
__global__ void convertDataType(const size_t n, const float *in, T *out);


template<typename T>
static __inline__ __device__ T warp_reduce_sum(T value) {
    /* aggregate all value that each thread within a warp holding.*/
    T ret = value;

    for (int w = 1; w < warpSize; w = w << 1) {
        T tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}

// Only supports 1D blocks
template<typename T>
static __inline__ __device__ T reduce_sum(T value, T *shm) {
    unsigned int stride;
    T tmp = warp_reduce_sum(value); // perform warp shuffle first for less utilized shared memory

    const unsigned int warpId = threadIdx.x >> 5;
    const unsigned int laneId = threadIdx.x & 31;
    if (laneId == 0) {
        shm[warpId] = tmp;
    }
    __syncthreads();
    for (stride = blockDim.x / (2 * warpSize); stride >= 1; stride = stride >> 1) {
        if (warpId < stride && laneId == 0) {
            shm[warpId] += shm[warpId + stride];
        }

        __syncthreads();
    }
    return shm[0];
}

/**
 * @funcitonName: makeData
 * @functionInterpretation: Use cuRand library. Generate `size` float random numbers in GPU device memory `data` that follow uniform distribution in [0.0, 1.0).
 **/
void makeData(float *data, const int size);

/**
 * @funcitonName: printCudaErrorStringSync
 * @functionInterpretation: Print the error message of the cuda runtime API, and synchronize the device
 **/
inline void printCudaErrorStringSync(const std::string &msg = "") {
    if (!msg.empty()) {
        printf("%s. ", msg.c_str());
    }
    printf("CUDA Error: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}

/**
 * @funcitonName: calculateOccupancyMaxPotentialBlockSize
 * @functionInterpretation: Calculate the optimal block size of the kernel function
 * @input:
 *  `func` : Kernel function
 * @output:
 * return a pair of int, the first element is the optimal block size, the second element is the minimal grid size
 **/
inline std::pair<int, int> calculateOccupancyMaxPotentialBlockSize(void *func) {
    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       func,
                                       0,
                                       0);
    printf("minGridSize: %d, blockSize: %d\n", minGridSize, blockSize);
    return std::make_pair(blockSize, minGridSize);
}
} // namespace cuUtil