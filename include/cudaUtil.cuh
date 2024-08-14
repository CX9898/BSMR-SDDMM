#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include <vector>

inline void printCudaErrorStringSync() {
    fprintf(stderr, "CUDA Error : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}

template<typename T>
inline void H2D(T *dev, const T *host, const size_t size) {
    cudaMemcpy(dev, host, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(T *host, const T *dev, const size_t size) {
    cudaMemcpy(host, dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void H2D(T *dev, const std::vector<T> &host) {
    cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(std::vector<T> &host, const T *dev, const size_t size) {
    host.clear();
    host.resize(size);
    cudaMemcpy(host.data(), dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline std::vector<T> D2H(const T *dev, const size_t size) {
    std::vector<T> host(size);
    cudaMemcpy(host.data(), dev, size * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

template<typename T>
inline void D2D(T *dest, const T *src, const size_t size) {
    cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToHost);
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