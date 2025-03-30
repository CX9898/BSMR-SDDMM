#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/count.h>

#include "parallelAlgorithm.cuh"

struct IsPositive {
  __host__ __device__
  bool operator()(int x) {
      return x > 0;
  }
};

struct is_equal {
  int value;

  is_equal(int v) : value(v) {}

  __host__ __device__
  bool operator()(int x) const {
      return x == value;
  }
};

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
void sort_by_key(int *key_first, int *key_last, uint32_t *value_first) {
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
void sequence(int *first, int *last, int start_value, int step) {
    thrust::sequence(thrust::host, first, last, start_value, step);
}
void sequence(uint32_t *first, uint32_t *last, uint32_t start_value, uint32_t step) {
    thrust::sequence(thrust::host, first, last, start_value, step);
}
size_t count_if_positive(uint32_t *first, uint32_t *last) {
    return thrust::count_if(thrust::host, first, last, IsPositive());
}
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *result) {
    thrust::copy_if(thrust::host, first, last, result, IsPositive());
}
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *stencil, uint32_t *result) {
    thrust::copy_if(thrust::host, first, last, stencil, result, IsPositive());
}
void computeRowNNZCountsFromOffsets(size_t num, uint32_t *offsets, uint32_t *result) {
    thrust::transform(thrust::host, offsets + 1, offsets + num + 1, offsets, result, thrust::minus<int>());
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
void sequence(uint32_t *first, uint32_t *last, uint32_t start_value, uint32_t step) {
    thrust::sequence(thrust::device, first, last, start_value, step);
}
void sort_by_key_descending_order(uint32_t *key_first, uint32_t *key_last, uint32_t *value_first) {
    auto descending = thrust::greater<int>();
    thrust::sort_by_key(thrust::device, key_first, key_last, value_first, descending);
}
size_t count_if_positive(uint32_t *first, uint32_t *last) {
    return thrust::count_if(thrust::device, first, last, IsPositive());
}
size_t count_if_equal(uint32_t *first, uint32_t *last, uint32_t value) {
    return thrust::count_if(thrust::device, first, last, is_equal(value));
}
void copy(uint32_t *first, uint32_t *last, uint32_t *result) {
    thrust::copy(thrust::device, first, last, result);
}
void copy_if_positive(uint32_t *first, uint32_t *last, uint32_t *stencil, uint32_t *result) {
    thrust::copy_if(thrust::device, first, last, stencil, result, IsPositive());
}
} // namespace dev