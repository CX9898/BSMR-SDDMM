#pragma once

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "devVector.cuh"

namespace cuUtil {
inline void printCudaErrorStringSync() {
    fprintf(stderr, "CUDA Error : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}

namespace host {

void sort(uint64_t *first, uint64_t *last);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first);

} // namespace host

namespace dev {

void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first);

inline void cudaSync() {
    cudaDeviceSynchronize();
}

} // namespace dev

}// namespace cuUtil