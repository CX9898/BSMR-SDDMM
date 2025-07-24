#pragma once

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "devVector.cuh"

inline void printCudaErrorStringSync() {
    fprintf(stderr, "CUDA Error : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
inline void cudaSync() {
    cudaDeviceSynchronize();
}

namespace host {
void fill_n(uint32_t *first, size_t size, uint32_t val);
void sort(uint32_t *first, uint32_t *last);
void sort(uint64_t *first, uint64_t *last);
void sort_by_key(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first);
void sort_by_key(uint32_t *key_first, uint32_t *key_last, int *value_first);
void sort_by_key(uint32_t *key_first, uint32_t *key_last, float *value_first);
void sort_by_key(uint32_t *key_first, uint32_t *key_last, double *value_first);
void sort_by_key(int *key_first, int *key_last, uint32_t *value_first);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first);
void sort_by_key_descending_order(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first);
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      uint32_t *value2_first);
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      int *value2_first);
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      float *value2_first);
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      double *value2_first);
void inclusive_scan(size_t *first, size_t *last, size_t *result);
void inclusive_scan(uint32_t *first, uint32_t *last, uint32_t *result);
void sequence(int *first, int *last, int start_value, int step = 1);
void sequence(uint32_t *first, uint32_t *last, uint32_t start_value, uint32_t step = 1);
size_t count_if_positive(uint32_t *first, uint32_t *last);
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *result);
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *stencil, uint32_t *result);
void computeRowNNZCountsFromOffsets(size_t num, uint32_t *offsets, uint32_t *result);
} // namespace host

namespace dev {
void fill_n(uint32_t *first, size_t size, uint32_t val);
void sort(uint32_t *first, uint32_t *last);
void sort(uint64_t *first, uint64_t *last);
void sort_by_key(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first);
void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first);
void inclusive_scan(size_t *first, size_t *last, size_t *result);
void inclusive_scan(uint32_t *first, uint32_t *last, uint32_t *result);
void sequence(uint32_t *first, uint32_t *last, uint32_t start_value, uint32_t step = 1);
void sort_by_key_descending_order(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first);
size_t count_if_positive(uint32_t *first, uint32_t *last);
size_t count_if_equal(uint32_t *first, uint32_t *last, uint32_t value);
void copy(uint32_t *first, uint32_t *last, uint32_t* result);
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *stencil, uint32_t *result);
} // namespace dev