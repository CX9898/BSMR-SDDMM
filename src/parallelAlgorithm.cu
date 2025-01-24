#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

#include "parallelAlgorithm.cuh"

namespace host {
void fill_n(uint32_t *first, size_t size, uint32_t val) {
    thrust::fill_n(thrust::host, first, size, val);
}
void sort(uint32_t *first, uint32_t *last) {
    thrust::sort(thrust::host, first, last);
}
void sort(uint64_t *first, uint64_t *last) {
    thrust::sort(thrust::host, first, last);
}
void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key(uint32_t *key_first, uint32_t *key_last, int *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key(uint32_t *key_first, uint32_t *key_last, float *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key(uint32_t *key_first, uint32_t *key_last, double *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
void sort_by_key_descending_order(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first) {
    auto descending = thrust::greater<int>();
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first, descending);
}
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      uint32_t *value2_first) {
    thrust::sort_by_key(thrust::host,
                        key_first,
                        key_last,
                        thrust::make_zip_iterator(thrust::make_tuple(value1_first, value2_first)));

}
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      int *value2_first) {
    thrust::sort_by_key(thrust::host,
                        key_first,
                        key_last,
                        thrust::make_zip_iterator(thrust::make_tuple(value1_first, value2_first)));

}
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      float *value2_first) {
    thrust::sort_by_key(thrust::host,
                        key_first,
                        key_last,
                        thrust::make_zip_iterator(thrust::make_tuple(value1_first, value2_first)));

}
void sort_by_key_for_multiple_vectors(uint32_t *key_first,
                                      uint32_t *key_last,
                                      uint32_t *value1_first,
                                      double *value2_first) {
    thrust::sort_by_key(thrust::host,
                        key_first,
                        key_last,
                        thrust::make_zip_iterator(thrust::make_tuple(value1_first, value2_first)));

}
void inclusive_scan(size_t *first, size_t *last, size_t *result) {
    thrust::inclusive_scan(thrust::host, first, last, result);
}
void inclusive_scan(uint32_t *first, uint32_t *last, uint32_t *result) {
    thrust::inclusive_scan(thrust::host, first, last, result);
}
} // namespace host

namespace dev {
void fill_n(uint32_t *first, size_t size, uint32_t val) {
    thrust::fill_n(thrust::device, first, size, val);
}
void sort(uint32_t *first, uint32_t *last) {
    thrust::sort(thrust::device, first, last);
}
void sort(uint64_t *first, uint64_t *last) {
    thrust::sort(thrust::device, first, last);
}
void sort_by_key(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first) {
    thrust::sort_by_key(thrust::device, key_first, key_last, value_first);
}
void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first) {
    thrust::sort_by_key(thrust::device, key_first, key_last, value_first);
}
void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first) {
    thrust::sort_by_key(thrust::device, key_first, key_last, value_first);
}
void inclusive_scan(size_t *first, size_t *last, size_t *result) {
    thrust::inclusive_scan(thrust::device, first, last, result);
}
void inclusive_scan(uint32_t *first, uint32_t *last, uint32_t *result) {
    thrust::inclusive_scan(thrust::device, first, last, result);
}
} // namespace dev