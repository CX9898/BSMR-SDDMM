#include <thrust/sort.h>

#include "cudaUtil.cuh"

namespace cuUtil {

namespace host {

void sort(uint64_t *first, uint64_t *last) {
    thrust::sort(thrust::host, first, last);
}

void sort_by_key(uint64_t *key_first, uint64_t *key_last, uint64_t *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}

void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first) {
    thrust::sort_by_key(thrust::host, key_first, key_last, value_first);
}
} // namespace host

namespace dev {

void sort_by_key(uint64_t *key_first, uint64_t *key_last, float *value_first) {
    thrust::sort_by_key(thrust::device, key_first, key_last, value_first);
}
} // namespace dev

}// namespace cuUtil