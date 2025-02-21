#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace cuUtil {

/**
 * @funcitonName: printCudaErrorStringSync
 * @functionInterpretation: Print the error message of the cuda runtime API, and synchronize the device
 **/
inline void printCudaErrorStringSync() {
    printf("cuda error : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
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