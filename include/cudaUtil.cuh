#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace cuUtil {
inline void printCudaErrorStringSync() {
    printf("cuda error : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
} // namespace cuUtil